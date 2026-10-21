"""Which providers handle Hebrew best, measured.

Run from the repo root with the keys in Windows Credential Manager:

    .venv\\Scripts\\python tools\\bench_hebrew.py

Three rounds, each printing a table:

1. Voice (TTS): every candidate speaks the same Hebrew sentences; each clip
   is transcribed by a strong Hebrew recognizer and compared with the text.
   A voice that comes back close to what it was given is intelligible; the
   time to first audio byte is the latency the user hears.
2. Listening (STT): the clips from the best voice are transcribed by every
   recognizer; closeness to the text and time per clip.
3. Model (LLM): a screenshot and a Hebrew question go to each chat model
   with Clicky's own system prompt; time to first token, whether the answer
   is Hebrew, and whether the [POINT] tag parsed.

Similarity is difflib's ratio on normalized text (no punctuation, no
niqqud), 0..1. Clips are written to the scratch folder given by --out.
"""
from __future__ import annotations

import argparse
import difflib
import io
import os
import re
import sys
import time
import wave
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import keyring  # noqa: E402

SERVICE = "clicky-windows"

SENTENCES = [
    "לחץ על כפתור השמירה בפינה הימנית העליונה של המסך.",
    "פתח את ההגדרות, עבור ללשונית פרטיות ואפשר את המצלמה.",
    "המדפסת לא מדפיסה כי תור ההדפסה תקוע, נפעיל מחדש את השירות.",
]

_NIQQUD = re.compile(r"[֑-ׇ]")
_PUNCT = re.compile(r"[^\w\s]")


def norm(text: str) -> str:
    text = _NIQQUD.sub("", text)
    text = _PUNCT.sub(" ", text)
    return " ".join(text.split()).lower()


def similarity(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, norm(a), norm(b)).ratio()


def key(name: str) -> str | None:
    return os.getenv(name) or keyring.get_password(SERVICE, name)


def pcm16_to_wav(pcm: bytes, rate: int) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm)
    return buf.getvalue()


# --- TTS candidates: each returns (wav_bytes, first_audio_seconds) ------------

def tts_elevenlabs(model_id: str):
    from elevenlabs import ElevenLabs
    client = ElevenLabs(api_key=key("ELEVENLABS_API_KEY"))

    def speak(text: str):
        t0 = time.perf_counter()
        first = None
        chunks = []
        for chunk in client.text_to_speech.stream(
            text=text, voice_id=os.getenv("ELEVENLABS_VOICE_ID", "21m00Tcm4TlvDq8ikWAM"),
            model_id=model_id, output_format="pcm_22050", language_code="he",
        ):
            if chunk:
                if first is None:
                    first = time.perf_counter() - t0
                chunks.append(chunk)
        return pcm16_to_wav(b"".join(chunks), 22050), first or 0.0
    return speak


def tts_openai(model: str, voice: str = "alloy"):
    from openai import OpenAI
    client = OpenAI(api_key=key("OPENAI_API_KEY"))

    def speak(text: str):
        t0 = time.perf_counter()
        first = None
        chunks = []
        with client.audio.speech.with_streaming_response.create(
            model=model, voice=voice, input=text, response_format="pcm",
        ) as resp:
            for chunk in resp.iter_bytes(4096):
                if chunk:
                    if first is None:
                        first = time.perf_counter() - t0
                    chunks.append(chunk)
        return pcm16_to_wav(b"".join(chunks), 24000), first or 0.0
    return speak


# --- STT candidates: each takes wav bytes, returns (text, seconds) -----------

def stt_openai(model: str):
    from openai import OpenAI
    client = OpenAI(api_key=key("OPENAI_API_KEY"))

    def listen(wav: bytes):
        t0 = time.perf_counter()
        f = io.BytesIO(wav); f.name = "clip.wav"
        r = client.audio.transcriptions.create(model=model, file=f, language="he")
        return r.text, time.perf_counter() - t0
    return listen


def stt_faster_whisper(size: str):
    from faster_whisper import WhisperModel
    import numpy as np
    model = WhisperModel(size, device="cpu", compute_type="int8")

    def listen(wav: bytes):
        with wave.open(io.BytesIO(wav)) as w:
            rate = w.getframerate()
            pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0
        if rate != 16000:
            idx = np.arange(0, len(pcm), rate / 16000)
            pcm = np.interp(idx, np.arange(len(pcm)), pcm).astype(np.float32)
        t0 = time.perf_counter()
        segs, _ = model.transcribe(pcm, language="he", beam_size=1)
        return " ".join(s.text for s in segs).strip(), time.perf_counter() - t0
    return listen


def stt_assemblyai_batch():
    import assemblyai as aai
    aai.settings.api_key = key("ASSEMBLYAI_API_KEY")

    def listen(wav: bytes):
        t0 = time.perf_counter()
        cfg = aai.TranscriptionConfig(language_code="he")
        tr = aai.Transcriber(config=cfg).transcribe(io.BytesIO(wav))
        return (tr.text or ""), time.perf_counter() - t0
    return listen


# --- LLM candidates ----------------------------------------------------------

def llm_round(models: list[str], image_path: Path | None):
    from PIL import Image, ImageDraw
    from ai import OpenAIVisionClient, build_system_prompt

    if image_path and image_path.is_file():
        img = Image.open(image_path).convert("RGB")
    else:
        img = Image.new("RGB", (1280, 800), "white")
        d = ImageDraw.Draw(img)
        d.rectangle((40, 20, 160, 60), fill="#dddddd"); d.text((60, 32), "File", fill="black")
        d.rectangle((180, 20, 300, 60), fill="#dddddd"); d.text((200, 32), "Save", fill="black")
    label = f"primary focus (image dimensions: {img.width}x{img.height} pixels)"
    prompt = build_system_prompt(language="he", step_mode=True, actions_mode="confirm")
    question = "איך שומרים את הקובץ? תראה לי איפה ללחוץ"
    rows = []
    for m in models:
        client = OpenAIVisionClient(api_key=key("OPENAI_API_KEY"), model_id=f"openai/{m}")
        t0 = time.perf_counter(); first = None; text = ""
        try:
            with client.ask_stream(images=[(img, label)], transcript=question, history=[], system_prompt=prompt) as s:
                for d in s.text_deltas():
                    if first is None:
                        first = time.perf_counter() - t0
                    text += d
                r = s.final_result()
            total = time.perf_counter() - t0
            hebrew = sum("֐" <= ch <= "׿" for ch in r.spoken_text) / max(1, len(r.spoken_text))
            rows.append((m, first or 0, total, round(hebrew, 2), r.coordinate is not None, r.spoken_text[:70]))
        except Exception as exc:
            rows.append((m, 0, 0, 0, False, f"ERROR {type(exc).__name__}: {str(exc)[:60]}"))
    print("\nLLM (Hebrew question + screenshot)")
    print(f"{'model':16s} {'first':>6s} {'total':>6s} {'he%':>5s} {'point':>5s}  answer")
    for m, f, t, h, p, a in rows:
        print(f"{m:16s} {f:6.2f} {t:6.2f} {h:5.2f} {str(p):>5s}  {a}")
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(Path.home() / ".clicky-windows" / "bench-hebrew"))
    ap.add_argument("--image", default="")
    ap.add_argument("--skip-local", action="store_true")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    judge = stt_openai("gpt-4o-transcribe")

    tts = {}
    if key("ELEVENLABS_API_KEY"):
        for m in ("eleven_flash_v2_5", "eleven_turbo_v2_5", "eleven_multilingual_v2", "eleven_v3"):
            tts[f"elevenlabs:{m}"] = tts_elevenlabs(m)
    if key("OPENAI_API_KEY"):
        tts["openai:gpt-4o-mini-tts"] = tts_openai("gpt-4o-mini-tts")
        tts["openai:tts-1"] = tts_openai("tts-1")

    print("TTS (Hebrew sentences; judged by gpt-4o-transcribe)")
    print(f"{'voice':32s} {'first':>6s} {'sim':>5s}")
    clips: dict[str, list[bytes]] = {}
    tts_scores = {}
    for name, speak in tts.items():
        firsts, sims, wavs = [], [], []
        for i, s in enumerate(SENTENCES):
            try:
                wav, first = speak(s)
            except Exception as exc:
                print(f"{name:32s} ERROR {type(exc).__name__}: {str(exc)[:80]}")
                break
            (out / f"{name.replace(':', '_')}_{i}.wav").write_bytes(wav)
            heard, _ = judge(wav)
            firsts.append(first); sims.append(similarity(s, heard)); wavs.append(wav)
        else:
            clips[name] = wavs
            tts_scores[name] = (sum(sims) / len(sims), sum(firsts) / len(firsts))
            print(f"{name:32s} {tts_scores[name][1]:6.2f} {tts_scores[name][0]:5.2f}")
    if not tts_scores:
        print("no voice produced Hebrew clips; stopping")
        return
    # Intelligibility first (within 0.05 counts as equal), then latency.
    best_tts = max(tts_scores, key=lambda n: (round(tts_scores[n][0], 1), -tts_scores[n][1]))
    print(f"best voice: {best_tts}")

    stt = {}
    if key("OPENAI_API_KEY"):
        for m in ("gpt-4o-transcribe", "gpt-4o-mini-transcribe", "whisper-1"):
            stt[f"openai:{m}"] = stt_openai(m)
    if key("ASSEMBLYAI_API_KEY"):
        stt["assemblyai:batch"] = stt_assemblyai_batch()
    if not args.skip_local:
        for size in ("small", "medium"):
            stt[f"faster-whisper:{size}"] = stt_faster_whisper(size)

    print(f"\nSTT (clips from {best_tts})")
    print(f"{'recognizer':28s} {'sec':>5s} {'sim':>5s}")
    stt_scores = {}
    for name, listen in stt.items():
        secs, sims = [], []
        try:
            for s, wav in zip(SENTENCES, clips[best_tts]):
                heard, sec = listen(wav)
                secs.append(sec); sims.append(similarity(s, heard))
        except Exception as exc:
            print(f"{name:28s} ERROR {type(exc).__name__}: {str(exc)[:80]}")
            continue
        stt_scores[name] = (sum(sims) / len(sims), sum(secs) / len(secs))
        print(f"{name:28s} {stt_scores[name][1]:5.2f} {stt_scores[name][0]:5.2f}")
    if stt_scores:
        best_stt = max(stt_scores, key=lambda n: (round(stt_scores[n][0], 1), -stt_scores[n][1]))
        print(f"best listening: {best_stt}")

    if key("OPENAI_API_KEY"):
        llm_round(["gpt-6.1-sol", "gpt-6-sol", "gpt-6-luna", "gpt-6-astra", "gpt-5.5"], Path(args.image) if args.image else None)


if __name__ == "__main__":
    main()
