"""Curated knowledge base: the user's own docs, read from ``KB_DIR`` and
every folder under it.

Two kinds of file, both plain Markdown the user can edit:

- ``<app_name>.md`` anywhere under ``KB_DIR`` (matched by sanitized .exe
  basename, like memory.py) is the reference for that one app. It comes
  first and is never dropped.
- Every other ``.md`` under ``KB_DIR`` is general knowledge (how-to guides,
  process catalogs). A folder named after a language (``he``, ``en``) holds
  that language's copy: the current LANGUAGE's folders are read, the other
  languages' are skipped. The whole tree can be far bigger than the budget
  (``KB_RECALL_MAX_CHARS``), so the files are ranked against the question:
  index files (``README.md``) first, then by how many of the question's words
  they contain, and whole files are added in that order until the budget is
  used. The model also gets each file's path, so it knows what else exists.

If nothing matches, ``recall()`` returns ``('', '')`` and the pipeline skips
KB injection (the "Claude already knows that software" path).
"""
from __future__ import annotations

import re
from pathlib import Path

from config import KB_DIR, KB_RECALL_MAX_CHARS, LANGUAGE

# Folder names that mark a language's copy of the docs.
_LANGUAGE_FOLDERS = frozenset({"en", "he"})
_WORD_RE = re.compile(r"[\w']+")
# Words too common to tell files apart; the rest of the question does that.
_STOP_WORDS = frozenset({
    "the", "a", "an", "and", "or", "to", "of", "in", "on", "is", "it", "i", "my",
    "me", "how", "do", "does", "can", "what", "where", "this", "that", "for",
    "with", "not", "no", "yes", "please", "את", "של", "על", "עם", "לא", "כן",
    "איך", "מה", "איפה", "זה", "יש", "לי", "אני", "אפשר", "בבקשה", "או", "גם",
    "אתה", "את", "יכול", "יכולה", "לעשות", "רוצה", "צריך", "צריכה", "אם", "כל",
    "הוא", "היא", "הם", "אבל", "אז", "רק", "עוד", "כאן", "שם", "לך", "לנו",
    "you", "your", "we", "our", "be", "are", "was", "have", "has", "will",
    "want", "need", "should", "would", "could", "just", "now", "here", "there",
})
_MIN_HITS = 2
"""Words a file must share with the question to be sent. One shared word is
usually a common one ("printer" in an index of every guide); two mean the file
is about the thing asked."""


def _sanitize_app_name(app_name: str) -> str:
    """Lowercase + replace filesystem-reserved chars.

    Mirrors ``memory._sanitize_app_name`` exactly so the same
    foreground app yields the same filename in both folders. Users
    can `ls ~/.clicky-windows/memory/` to learn the canonical name
    for their app, then drop the matching file in KB_DIR.
    """
    sanitized = app_name.lower()
    for ch in ":\\/":
        sanitized = sanitized.replace(ch, "_")
    return sanitized


def _words(text: str) -> set[str]:
    return {w for w in (m.lower() for m in _WORD_RE.findall(text)) if len(w) > 2 and w not in _STOP_WORDS}


def _in_language(path: Path, base: Path, language: str) -> bool:
    """A file is read unless it sits under another language's folder."""
    for part in path.relative_to(base).parts[:-1]:
        if part.lower() in _LANGUAGE_FOLDERS and part.lower() != language:
            return False
    return True


_cache: dict[Path, tuple[float, str, set[str]]] = {}
"""(mtime, text, words) per file. The folder can hold half a megabyte; reading
and tokenizing it every turn cost 0.6 s before the model was even asked. A
file is re-read when its modification time changes, so edits still count."""


def _read(path: Path) -> str:
    return _load(path)[0]


def _load(path: Path) -> tuple[str, set[str]]:
    try:
        mtime = path.stat().st_mtime
    except OSError:
        return "", set()
    hit = _cache.get(path)
    if hit and hit[0] == mtime:
        return hit[1], hit[2]
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return "", set()
    words = _words(text)
    _cache[path] = (mtime, text, words)
    return text, words


def recall(
    app_name: str,
    kb_dir: Path | None = None,
    max_chars: int = KB_RECALL_MAX_CHARS,
    query: str = "",
    language: str | None = None,
) -> tuple[str, str]:
    """Collect the knowledge for this question.

    Args:
        app_name: foreground .exe basename, e.g. ``"EDUPACK.EXE"``.
        kb_dir: override the KB folder (test hook). Defaults to
            ``config.KB_DIR``.
        max_chars: budget for everything returned. Defaults to
            ``config.KB_RECALL_MAX_CHARS``.
        query: the user's question, used to rank the general docs.
        language: which language folders to read (``config.LANGUAGE``).

    Returns:
        ``(content, name)``. ``content`` is the app file (tail-truncated to
        the budget on its own, as before) followed by the best-matching
        general docs, each under a ``### <relative path>`` heading. ``name``
        is the sanitized app name when its file was found (``ai.py`` shows
        it in the injection marker), else ``"the knowledge folder"``.

        Both empty if nothing matches, ``app_name`` is blank, or
        ``max_chars <= 0`` (defensive guard: Python's ``text[-0:]``
        otherwise returns the whole string).
    """
    if max_chars <= 0:
        return ("", "")
    if not app_name or not app_name.strip():
        return ("", "")

    base_dir = Path(kb_dir) if kb_dir is not None else KB_DIR
    if not base_dir.is_dir():
        return ("", "")
    language = (language or LANGUAGE).lower()
    sanitized = _sanitize_app_name(app_name)
    app_file = f"{sanitized}.md"

    files = sorted(p for p in base_dir.rglob("*.md") if p.is_file())
    app_paths = [p for p in files if p.name.lower() == app_file]
    general = [p for p in files if p.name.lower() != app_file and _in_language(p, base_dir, language)]

    parts: list[str] = []
    budget = max_chars
    name = ""
    if app_paths:
        text = _read(app_paths[0])
        if len(text) > budget:
            text = text[-budget:]
        parts.append(text)
        budget -= len(text)
        name = sanitized

    if general and budget > 0:
        asked = _words(query)
        ranked = []
        for p in general:
            text, words = _load(p)
            if not text.strip():
                continue
            rel = p.relative_to(base_dir).as_posix()
            hits = len(asked & words) if asked else 0
            is_index = p.name.lower() == "readme.md"
            depth = len(p.relative_to(base_dir).parts)
            # Only what the question touches: a file is sent when it shares a
            # word with the question, plus the top indexes (two levels down
            # at most) so the model knows what else the folder holds. A
            # question about nothing in the folder costs a few hundred tokens
            # instead of the whole budget. Indexes first, then the files
            # sharing the most words, shortest first among equals.
            if hits < _MIN_HITS and not (is_index and depth <= 2):
                continue
            ranked.append(((0 if is_index else 1, -hits, len(text)), rel, text))
        ranked.sort(key=lambda item: item[0])
        for _, rel, text in ranked:
            block = f"\n\n### {rel}\n{text.strip()}"
            if len(block) > budget:
                continue  # whole files only, so a guide is never cut mid-step
            parts.append(block)
            budget -= len(block)
        if len(parts) > (1 if app_paths else 0) and not name:
            name = "the knowledge folder"

    if not parts:
        return ("", "")
    return ("".join(parts), name)


if __name__ == "__main__":
    print("=" * 70)
    print("Clicky Windows -- kb.py manual verification")
    print(f"  KB_DIR: {KB_DIR}")
    print(f"  KB_RECALL_MAX_CHARS: {KB_RECALL_MAX_CHARS}")
    print(f"  LANGUAGE: {LANGUAGE}")
    print("=" * 70)

    for app, q in [("EDUPACK.EXE", ""), ("outlook.exe", "the printer does not print"), ("DOES_NOT_EXIST.EXE", "")]:
        content, name = recall(app, query=q)
        if content:
            files = re.findall(r"^### (.+)$", content, re.MULTILINE)
            print(f"\n{app:24s} {q!r:34s} -> {name!r}, {len(content)} chars, files: {files[:6]}")
        else:
            print(f"\n{app:24s} {q!r:34s} -> nothing under {KB_DIR}")
