"""Guided steps, acting on the user's behalf, Hebrew and Opus 5.5.

Covers the pieces added across ai.py (prompt sections, Opus effort), hotkey.py
(the Ctrl+Alt+Enter confirm instance) and app.py (confirm flow, pending
actions, follow-up turns).
"""
from __future__ import annotations

import threading
from unittest.mock import MagicMock

import pytest


# --- ai.py -------------------------------------------------------------------

class TestOpus55Effort:
    """Opus 5.5 can't turn thinking off; low effort keeps the first spoken
    word quick. Sent as extra_body so any SDK version passes it through."""

    def _kwargs(self, mocker, model_id):
        from PIL import Image
        from ai import create_ai_client
        mock = mocker.patch("ai.Anthropic")
        client = create_ai_client(model_id=model_id, api_key="sk-ant-test")
        client.ask_stream(images=[(Image.new("RGB", (8, 8)), "s")], transcript="hi", history=[])
        return mock.return_value.messages.stream.call_args.kwargs

    def test_opus_5_5_runs_at_low_effort(self, mocker):
        kwargs = self._kwargs(mocker, "anthropic/claude-opus-5-5")
        assert kwargs["extra_body"] == {"output_config": {"effort": "low"}}
        assert "thinking" not in kwargs
        assert kwargs["model"] == "claude-opus-5-5"

    def test_haiku_sends_no_effort(self, mocker):
        assert "extra_body" not in self._kwargs(mocker, "anthropic/claude-haiku-5-5")


class TestBuildSystemPrompt:
    def test_defaults_are_the_plain_prompts(self):
        from ai import build_system_prompt, _CLICKY_SYSTEM_PROMPT, _CLICKY_ANNOTATION_SYSTEM_PROMPT
        assert build_system_prompt() == _CLICKY_SYSTEM_PROMPT
        assert build_system_prompt(annotation=True) == _CLICKY_ANNOTATION_SYSTEM_PROMPT

    def test_hebrew_section_asks_for_hebrew_and_english_tags(self):
        from ai import build_system_prompt, _CLICKY_SYSTEM_PROMPT
        prompt = build_system_prompt(language="he")
        assert prompt.startswith(_CLICKY_SYSTEM_PROMPT)
        extra = prompt[len(_CLICKY_SYSTEM_PROMPT):]
        assert "answer in hebrew" in extra and "[POINT]" in extra

    def test_unknown_language_adds_nothing(self):
        from ai import build_system_prompt, _CLICKY_SYSTEM_PROMPT
        assert build_system_prompt(language="fr") == _CLICKY_SYSTEM_PROMPT

    def test_step_mode_explains_step_and_done_tags(self):
        from ai import build_system_prompt
        prompt = build_system_prompt(step_mode=True)
        assert "[STEP]" in prompt and "[DONE]" in prompt and "control alt enter" in prompt

    def test_actions_confirm_vs_auto_notes(self):
        from ai import build_system_prompt
        confirm = build_system_prompt(actions_mode="confirm")
        auto = build_system_prompt(actions_mode="auto")
        for p in (confirm, auto):
            assert "[CLICK:x,y]" in p and "[POWERSHELL:command]" in p and "[OPEN:target]" in p
            assert "new screenshot of all the user's screens" in p
        assert "nothing runs until the user presses control alt enter" in confirm
        assert "run as soon as you have answered" in auto
        assert build_system_prompt(actions_mode="off") == build_system_prompt()

    def test_sections_stack_in_order(self):
        from ai import build_system_prompt
        prompt = build_system_prompt(language="he", step_mode=True, actions_mode="confirm")
        assert prompt.index("guided steps:") < prompt.index("acting for the user:") < prompt.index("language:")


class TestOllamaBoxPromptWithExtras:
    def test_cursor_prompt_with_sections_still_asks_for_a_box(self):
        """build_system_prompt output starts with the cursor prompt, so a local
        Qwen model still gets its box request, and keeps the extra sections."""
        from PIL import Image
        from ai import OllamaClient, build_system_prompt, _CLICKY_BOX_SYSTEM_PROMPT
        client = OllamaClient(host="http://localhost:11434", model_id="ollama/qwen3-vl:4b-instruct")
        stream = client.ask_stream(
            images=[(Image.new("RGB", (1280, 800)), "a")], transcript="where is save?",
            history=[], system_prompt=build_system_prompt(language="he", step_mode=True),
        )
        system = stream._payload["messages"][0]["content"]
        assert system.startswith(_CLICKY_BOX_SYSTEM_PROMPT)
        assert "answer in hebrew" in system and "guided steps:" in system
        assert "bbox_2d" in stream._payload["messages"][-1]["content"]
        assert stream._box_style == "grid"


# --- hotkey.py ---------------------------------------------------------------

class TestTriggerKey:
    """The confirm hotkey is the same class with Enter in place of Space."""

    def test_ctrl_alt_enter_fires_with_enter_trigger(self):
        from pynput import keyboard
        from hotkey import PushToTalkHotkey
        on_press, on_release = MagicMock(), MagicMock()
        hk = PushToTalkHotkey(on_press=on_press, on_release=on_release,
                              listener_class=MagicMock(), trigger_key=keyboard.Key.enter)
        hk._handle_press(keyboard.Key.ctrl_l)
        hk._handle_press(keyboard.Key.alt_l)
        hk._handle_press(keyboard.Key.space)
        on_press.assert_not_called()  # space is not this instance's trigger
        hk._handle_press(keyboard.Key.enter)
        on_press.assert_called_once()
        hk._handle_release(keyboard.Key.enter)
        on_release.assert_called_once()


# --- app.py ------------------------------------------------------------------

class TestConfirmation:
    @pytest.mark.parametrize("text", ["ok", "next", "Done.", "ok next", "I did it", "go ahead", "הבא", "סיימתי", "כן קדימה"])
    def test_short_confirmations(self, text):
        from app import _is_confirmation
        assert _is_confirmation(text) is True

    @pytest.mark.parametrize("text", ["", "ok where is the save button", "what is html", "next tab please open it now"])
    def test_questions_are_not_confirmations(self, text):
        from app import _is_confirmation
        assert _is_confirmation(text) is False

    def test_follow_up_note_carries_results(self):
        from app import _follow_up_note
        note = _follow_up_note(["click at (1, 2): done"], confirmed=True)
        assert "the user confirmed" in note and "click at (1, 2): done" in note
        assert "the user confirmed" not in _follow_up_note([], confirmed=False)


class TestConfirmFlow:
    @pytest.fixture(autouse=True)
    def _no_settle(self, mocker):
        mocker.patch("app._ACTION_SETTLE_S", 0)

    def _make_app(self, mocker):
        from app import ClickyApp
        return ClickyApp(
            ai_client=mocker.MagicMock(), stt_client=mocker.MagicMock(),
            tts_client=mocker.MagicMock(), memory_store=mocker.MagicMock(),
            overlay_controller=mocker.MagicMock(), hotkey_instance=mocker.MagicMock(),
            confirm_hotkey_instance=mocker.MagicMock(), mouse_listener_instance=mocker.MagicMock(),
        )

    def _cap(self, mocker):
        from PIL import Image
        cap = mocker.MagicMock()
        cap.image = Image.new("RGB", (1280, 800))
        cap.label = "screen 1 of 1"
        cap.scale_x = 2.0; cap.scale_y = 2.0
        cap.monitor = {"left": 0, "top": 0, "width": 2560, "height": 1600}
        cap.target_width = 1280; cap.target_height = 800
        cap.is_cursor_screen = True
        return cap

    def test_start_starts_both_hotkeys(self, mocker):
        mocker.patch("app.WATCH_MODE", "on")
        app = self._make_app(mocker)
        app.start()
        app._hotkey.start.assert_called_once()
        app._confirm_hotkey.start.assert_called_once()
        app._mouse_listener.start.assert_called_once()
        app.stop()
        app._confirm_hotkey.stop.assert_called_once()
        app._mouse_listener.stop.assert_called_once()

    def test_stray_confirm_with_nothing_asked_is_ignored(self, mocker):
        app = self._make_app(mocker)
        spawn = mocker.patch.object(app, "_spawn_pipeline")
        app._handle_confirm()
        app._tts.stop.assert_not_called()
        spawn.assert_not_called()

    def _run_threads_inline(self, mocker):
        """The confirm worker runs on a thread and hands back through a Qt
        signal, which needs an event loop; run it inline instead."""
        class _Inline:
            def __init__(self, target, **_kw):
                self._target = target
            def start(self):
                self._target()
        mocker.patch("app.threading.Thread", _Inline)

    def test_confirm_runs_pending_actions_then_follows_up(self, mocker):
        from actions import Click
        app = self._make_app(mocker)
        self._run_threads_inline(mocker)
        app._history = [{"role": "user", "content": []}]
        app._pending_actions = [Click(100, 50)]
        app._action_capture = self._cap(mocker)
        clicker = mocker.patch("actions.click_at")
        got = threading.Event()
        notes = []
        app.sig_follow_up.disconnect(app._spawn_pipeline)
        app.sig_follow_up.connect(lambda n: (notes.append(n), got.set()))
        app._handle_confirm()
        assert got.wait(2)
        clicker.assert_called_once_with(200, 100)  # 2x scale, no monitor offset
        assert app._pending_actions == []
        assert "the user confirmed" in notes[0] and "click at (100, 50): done" in notes[0]
        app._tts.stop.assert_called_once()

    def test_confirm_after_a_step_follows_up_without_actions(self, mocker):
        app = self._make_app(mocker)
        self._run_threads_inline(mocker)
        app._history = [{"role": "user", "content": []}]
        got = threading.Event()
        app.sig_follow_up.disconnect(app._spawn_pipeline)
        app.sig_follow_up.connect(lambda n: got.set())
        app._handle_confirm()
        assert got.wait(2)

    def test_press_drops_pending_actions(self, mocker):
        from actions import Click
        app = self._make_app(mocker)
        app._pending_actions = [Click(1, 1)]
        mocker.patch("app.get_foreground_app", return_value=("X.EXE", ""))
        app._handle_press()
        assert app._pending_actions == []

    def _run_pipeline(self, mocker, app, deltas, spoken, transcript=None):
        import queue as _queue
        cap = self._cap(mocker)
        fake_stream = mocker.MagicMock()
        fake_stream.text_deltas.return_value = iter(deltas)
        fake_stream.final_result.return_value = mocker.MagicMock(
            spoken_text=spoken, coordinate=None, element_label=None, screen_number=None,
        )
        app._ai.ask_stream.return_value.__enter__ = mocker.MagicMock(return_value=fake_stream)
        app._ai.ask_stream.return_value.__exit__ = mocker.MagicMock(return_value=False)
        q: _queue.Queue = _queue.Queue(maxsize=1)
        q.put(([cap], "", "test"))
        app._pipeline_worker("TEST.EXE", "T", threading.Event(), q, transcript)
        return cap

    def test_follow_up_transcript_skips_listening(self, mocker):
        app = self._make_app(mocker)
        self._run_pipeline(mocker, app, ["next, click save. "], "next, click save.", transcript="[note] what is next?")
        app._stt.stop_recording.assert_not_called()
        assert app._ai.ask_stream.call_args.kwargs["transcript"].endswith("what is next?")
        assert app._history[-2]["content"][0]["text"] == "[note] what is next?"

    def test_voice_next_after_a_step_is_a_confirmation(self, mocker):
        app = self._make_app(mocker)
        app._history = [{"role": "user", "content": [{"type": "text", "text": "how do i save"}]}]
        app._stt.stop_recording.return_value = "ok next"
        self._run_pipeline(mocker, app, ["now click ok. "], "now click ok.")
        sent = app._ai.ask_stream.call_args.kwargs["transcript"]
        assert "the user confirmed" in sent and "ok next" not in sent

    def test_voice_next_with_no_history_is_a_question(self, mocker):
        app = self._make_app(mocker)
        app._stt.stop_recording.return_value = "next"
        self._run_pipeline(mocker, app, ["next what? "], "next what?")
        assert app._ai.ask_stream.call_args.kwargs["transcript"] == "next"

    def test_step_and_action_tags_are_never_spoken(self, mocker):
        app = self._make_app(mocker)
        app._stt.stop_recording.return_value = "save this for me"
        self._run_pipeline(
            mocker, app, ["i will press control s. ", "[KEYS:ctrl+s][STEP]"],
            "i will press control s. [KEYS:ctrl+s][STEP]",
        )
        said = " ".join(c.args[0] for c in app._tts.speak_sentence.call_args_list)
        assert "KEYS" not in said and "STEP" not in said
        assert app._history[-1]["content"][0]["text"] == "i will press control s."

    def test_confirm_mode_holds_actions_until_confirmed(self, mocker):
        from actions import Keys
        mocker.patch("app.ACTIONS_MODE", "confirm")
        app = self._make_app(mocker)
        app._stt.stop_recording.return_value = "save this for me"
        presser = mocker.patch("actions.press_keys")
        cap = self._run_pipeline(mocker, app, ["saving. [KEYS:ctrl+s]"], "saving. [KEYS:ctrl+s]")
        presser.assert_not_called()
        assert app._pending_actions == [Keys("ctrl+s")]
        assert app._action_capture is cap

    def test_auto_mode_runs_actions_and_follows_up(self, mocker):
        mocker.patch("app.ACTIONS_MODE", "auto")
        app = self._make_app(mocker)
        app._stt.stop_recording.return_value = "save this for me"
        presser = mocker.patch("actions.press_keys")
        notes = []
        app.sig_follow_up.disconnect(app._spawn_pipeline)
        app.sig_follow_up.connect(notes.append)
        self._run_pipeline(mocker, app, ["saving. [KEYS:ctrl+s][DONE]"], "saving. [KEYS:ctrl+s][DONE]")
        presser.assert_called_once_with("ctrl+s")
        assert app._pending_actions == []
        assert app._auto_turns == 1
        assert notes and "press ctrl+s: done" in notes[0]

    def test_auto_mode_still_asks_before_powershell(self, mocker):
        from actions import PowerShell
        mocker.patch("app.ACTIONS_MODE", "auto")
        app = self._make_app(mocker)
        app._stt.stop_recording.return_value = "list my downloads"
        shell = mocker.patch("actions.run_powershell")
        self._run_pipeline(mocker, app, ["listing. [POWERSHELL:dir]"], "listing. [POWERSHELL:dir]")
        shell.assert_not_called()
        assert app._pending_actions == [PowerShell("dir")]

    def test_actions_off_ignores_tags(self, mocker):
        mocker.patch("app.ACTIONS_MODE", "off")
        app = self._make_app(mocker)
        app._stt.stop_recording.return_value = "save this"
        presser = mocker.patch("actions.press_keys")
        self._run_pipeline(mocker, app, ["saving. [KEYS:ctrl+s]"], "saving. [KEYS:ctrl+s]")
        presser.assert_not_called()
        assert app._pending_actions == []

    @pytest.mark.parametrize("draw", ["off", "on"])
    def test_pipeline_sends_the_built_prompt(self, mocker, draw):
        from app import _SYSTEM_PROMPT_CURSOR, _SYSTEM_PROMPT_DRAW
        from ai import _CLICKY_SYSTEM_PROMPT, _CLICKY_ANNOTATION_SYSTEM_PROMPT
        mocker.patch("app.ANNOTATION_MODE", draw)
        app = self._make_app(mocker)
        app._stt.stop_recording.return_value = "hi"
        self._run_pipeline(mocker, app, ["hello. "], "hello.")
        sent = app._ai.ask_stream.call_args.kwargs["system_prompt"]
        if draw == "on":
            assert sent == _SYSTEM_PROMPT_DRAW and sent.startswith(_CLICKY_ANNOTATION_SYSTEM_PROMPT)
        else:
            assert sent == _SYSTEM_PROMPT_CURSOR and sent.startswith(_CLICKY_SYSTEM_PROMPT)


class TestLiveScreen:
    """A click by the user during a guided task is "I did that": after a short
    settle Clicky looks at the screen again, with no hotkey."""

    @pytest.fixture(autouse=True)
    def _fast(self, mocker):
        mocker.patch("app._ACTION_SETTLE_S", 0.01)
        mocker.patch("app.WATCH_MODE", "on")

    def _make_app(self, mocker):
        from app import ClickyApp
        return ClickyApp(
            ai_client=mocker.MagicMock(), stt_client=mocker.MagicMock(),
            tts_client=mocker.MagicMock(), memory_store=mocker.MagicMock(),
            overlay_controller=mocker.MagicMock(), hotkey_instance=mocker.MagicMock(),
            confirm_hotkey_instance=mocker.MagicMock(), mouse_listener_instance=mocker.MagicMock(),
        )

    def _left(self):
        from pynput import mouse
        return mouse.Button.left

    def test_click_during_a_step_looks_at_the_screen(self, mocker):
        app = self._make_app(mocker)
        app._step_pending = True
        spawn = mocker.patch.object(app, "_spawn_pipeline")
        app._on_mouse_click(10, 10, self._left(), pressed=True)
        app._on_mouse_click(10, 10, self._left(), pressed=False)
        app._click_timer.join(1)
        app._handle_user_click()  # what the queued signal does on the main thread
        spawn.assert_called_once()
        assert "the user just clicked" in spawn.call_args.args[0]
        app._tts.stop.assert_called_once()

    def test_no_step_outstanding_ignores_clicks(self, mocker):
        app = self._make_app(mocker)
        app._step_pending = False
        app._on_mouse_click(10, 10, self._left(), pressed=False)
        assert app._click_timer is None

    def test_clickys_own_clicks_and_pending_actions_do_not_count(self, mocker):
        from actions import Click
        app = self._make_app(mocker)
        app._step_pending = True
        app._running_actions = True
        app._on_mouse_click(1, 1, self._left(), pressed=False)
        assert app._click_timer is None
        app._running_actions = False
        app._pending_actions = [Click(1, 1)]
        app._on_mouse_click(1, 1, self._left(), pressed=False)
        assert app._click_timer is None

    def test_right_click_ignored(self, mocker):
        from pynput import mouse
        app = self._make_app(mocker)
        app._step_pending = True
        app._on_mouse_click(1, 1, mouse.Button.right, pressed=False)
        assert app._click_timer is None

    def test_watch_off_ignores_clicks(self, mocker):
        mocker.patch("app.WATCH_MODE", "off")
        app = self._make_app(mocker)
        app._step_pending = True
        app._on_mouse_click(1, 1, self._left(), pressed=False)
        assert app._click_timer is None

    def test_click_while_answering_is_retried_later(self, mocker):
        app = self._make_app(mocker)
        app._step_pending = True
        app._worker_thread = mocker.MagicMock(is_alive=lambda: True)
        spawn = mocker.patch.object(app, "_spawn_pipeline")
        app._handle_user_click()
        spawn.assert_not_called()
        assert app._click_timer is not None
        app._click_timer.cancel()

    def test_step_tag_sets_and_done_clears_the_task(self, mocker):
        import queue as _queue
        from PIL import Image
        app = self._make_app(mocker)
        cap = mocker.MagicMock(image=Image.new("RGB", (8, 8)), label="s", scale_x=1.0, scale_y=1.0,
                               monitor={"left": 0, "top": 0}, target_width=8, target_height=8)
        for spoken, expected in (("click file. [STEP]", True), ("all done. [DONE]", False), ("html is markup.", False)):
            stream = mocker.MagicMock()
            stream.text_deltas.return_value = iter([spoken])
            stream.final_result.return_value = mocker.MagicMock(spoken_text=spoken, coordinate=None, element_label=None, screen_number=None)
            app._ai.ask_stream.return_value.__enter__ = mocker.MagicMock(return_value=stream)
            app._ai.ask_stream.return_value.__exit__ = mocker.MagicMock(return_value=False)
            q = _queue.Queue(maxsize=1); q.put(([cap], "", "t"))
            app._stt.stop_recording.return_value = "how do i save"
            app._pipeline_worker("X.EXE", "", threading.Event(), q)
            assert app._step_pending is expected, spoken


# --- OpenAI listening and voice (the Hebrew defaults) ------------------------

class TestOpenAIProviders:
    def test_stt_factory_routes_openai(self):
        from stt import create_stt_client, OpenAISTT
        assert isinstance(create_stt_client("openai", "sk-test"), OpenAISTT)

    def test_tts_factory_routes_openai(self):
        from tts import create_tts_client, OpenAITTS
        assert isinstance(create_tts_client("openai", "sk-test"), OpenAITTS)

    def test_openai_stt_sends_wav_with_language(self, mocker):
        from stt import OpenAISTT
        client = MagicMock()
        client.audio.transcriptions.create.return_value = MagicMock(text=" שלום עולם ")
        stt = OpenAISTT(api_key="sk", client_factory=lambda k: client, audio_stream_factory=lambda cb: MagicMock(), language="he")
        stt.connect(); stt.start_recording()
        stt._buffer.extend(b"\x00\x01" * 1600)
        assert stt.stop_recording() == "שלום עולם"
        kwargs = client.audio.transcriptions.create.call_args.kwargs
        assert kwargs["language"] == "he" and kwargs["model"] == "gpt-4o-transcribe"
        assert kwargs["file"].read(4) == b"RIFF"

    def test_openai_stt_empty_buffer_is_silent(self):
        from stt import OpenAISTT
        stt = OpenAISTT(api_key="sk", client_factory=lambda k: MagicMock(), audio_stream_factory=lambda cb: MagicMock())
        stt.connect(); stt.start_recording()
        assert stt.stop_recording() == ""

    def test_openai_tts_streams_pcm_and_plays_float32(self):
        import numpy as np
        from tts import OpenAITTS
        resp = MagicMock()
        resp.iter_bytes.return_value = iter([np.array([0, 16384, -16384], dtype=np.int16).tobytes()])
        client = MagicMock()
        client.audio.speech.with_streaming_response.create.return_value.__enter__ = MagicMock(return_value=resp)
        played = []
        tts = OpenAITTS(api_key="sk", client_factory=lambda api_key: client, player_factory=lambda sample_rate: (played.append, None))
        tts.speak("שלום")
        tts._current_thread.join(3)
        kwargs = client.audio.speech.with_streaming_response.create.call_args.kwargs
        assert kwargs["model"] == "gpt-4o-mini-tts" and kwargs["response_format"] == "pcm" and kwargs["input"] == "שלום"
        assert len(played) == 1 and played[0].dtype == np.float32 and abs(played[0][1] - 0.5) < 1e-6
        resp.close.assert_called()

    def test_credentials_for_openai_speech_use_the_openai_key(self, mocker):
        import app
        mocker.patch("app.resolve_setting", side_effect=lambda n, default=None: "openai")
        mocker.patch("app.resolve_api_key", side_effect=lambda n: "sk-x" if n == "OPENAI_API_KEY" else None)
        assert app._resolve_tts_credentials() == ("openai", "sk-x")
        assert app._resolve_stt_credentials() == ("openai", "sk-x")
