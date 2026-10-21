"""Unit tests for kb.py.

Mock-free where possible: uses pytest's ``tmp_path`` fixture for real
filesystem round-trips. Mirrors test_memory.py style.
"""
from __future__ import annotations

from pathlib import Path

import pytest


def test_kb_module_importable():
    import kb  # noqa: F401


# --- TestSanitizeAppName -----------------------------------------------------

class TestSanitizeAppName:
    """Pure function tests: no fixture, no filesystem."""

    def test_sanitize_lowercases_keeping_dot_exe(self):
        from kb import _sanitize_app_name
        assert _sanitize_app_name("EXCEL.EXE") == "excel.exe"
        assert _sanitize_app_name("EduPack.EXE") == "edupack.exe"

    def test_sanitize_replaces_filesystem_reserved_chars(self):
        from kb import _sanitize_app_name
        assert _sanitize_app_name("My App: v2.EXE") == "my app_ v2.exe"
        assert _sanitize_app_name("a/b\\c") == "a_b_c"


# --- TestRecall --------------------------------------------------------------

class TestRecall:
    """recall() reads <app>.md from KB_DIR (or override). Returns
    (content, sanitized_name) or ('', '') for the empty paths."""

    def test_recall_returns_empty_for_missing_file(self, tmp_path: Path):
        """No .md file for this app → ('', ''): Claude already knows."""
        from kb import recall
        content, name = recall("edupack.exe", kb_dir=tmp_path)
        assert content == ""
        assert name == ""

    def test_recall_returns_empty_when_kb_dir_does_not_exist(
        self, tmp_path: Path
    ):
        """Override pointing at a missing dir → ('', ''), no exception."""
        from kb import recall
        missing = tmp_path / "does_not_exist"
        content, name = recall("edupack.exe", kb_dir=missing)
        assert content == ""
        assert name == ""

    def test_recall_returns_full_content_when_file_under_budget(
        self, tmp_path: Path
    ):
        """Small file (< max_chars) → full content + sanitized name."""
        from kb import recall
        body = "# EduPack KB\n\n## Plotting\nUse Chart > Add..."
        (tmp_path / "edupack.exe.md").write_text(body, encoding="utf-8")
        content, name = recall("EDUPACK.EXE", kb_dir=tmp_path)
        assert content == body
        assert name == "edupack.exe"

    def test_recall_tail_truncates_when_over_budget(self, tmp_path: Path):
        """File > max_chars → exactly the last max_chars chars."""
        from kb import recall
        big_body = "x" * 80_000
        (tmp_path / "fusion360.exe.md").write_text(big_body, encoding="utf-8")
        content, _ = recall("Fusion360.exe", kb_dir=tmp_path, max_chars=60_000)
        assert len(content) == 60_000
        assert content == big_body[-60_000:]

    def test_recall_returns_empty_for_non_positive_max_chars(
        self, tmp_path: Path
    ):
        """max_chars <= 0 → ('', '') (defensive: Python's text[-0:]
        returns FULL string, so a misconfigured caller would get
        the whole file. Fail closed.)."""
        from kb import recall
        (tmp_path / "edupack.exe.md").write_text("hello", encoding="utf-8")
        assert recall("edupack.exe", kb_dir=tmp_path, max_chars=0) == ("", "")
        assert recall("edupack.exe", kb_dir=tmp_path, max_chars=-5) == ("", "")
        # Sanity: positive max_chars works on the same file.
        assert recall("edupack.exe", kb_dir=tmp_path)[0] != ""

    def test_recall_returns_empty_for_blank_app_name(self, tmp_path: Path):
        """Empty/whitespace app_name → ('', ''), no exception."""
        from kb import recall
        assert recall("", kb_dir=tmp_path) == ("", "")
        assert recall("   ", kb_dir=tmp_path) == ("", "")

    def test_recall_uses_default_kb_dir_when_override_is_none(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """Without explicit kb_dir param, falls back to config.KB_DIR.

        Patches kb.KB_DIR module-level binding (the value imported at
        module load time). This is how tests verify the env-override
        path without touching the real ~/Documents/Clicky Wiki/.
        """
        from kb import recall
        import kb as kb_module
        monkeypatch.setattr(kb_module, "KB_DIR", tmp_path)
        body = "# Blender KB\n"
        (tmp_path / "blender.exe.md").write_text(body, encoding="utf-8")
        content, name = recall("blender.exe")
        assert content == body
        assert name == "blender.exe"


class TestKnowledgeFolder:
    """Every .md under KB_DIR and its subfolders is knowledge; the current
    language's folders are read, other languages' skipped; files are ranked
    against the question and added whole until the budget is used."""

    def _tree(self, tmp_path):
        (tmp_path / "en").mkdir(); (tmp_path / "he").mkdir(); (tmp_path / "cat" / "he").mkdir(parents=True)
        (tmp_path / "en" / "README.md").write_text("# guides\n1. camera\n2. printer", encoding="utf-8")
        (tmp_path / "en" / "01-camera.md").write_text("# camera not working\nopen settings, privacy, camera.", encoding="utf-8")
        (tmp_path / "en" / "03-printer.md").write_text("# printer does not print\ncheck the printer queue and restart the spooler.", encoding="utf-8")
        (tmp_path / "he" / "03-printer.md").write_text("# המדפסת לא מדפיסה\nבדוק את תור ההדפסה.", encoding="utf-8")
        (tmp_path / "cat" / "he" / "O2C.md").write_text("# הזמנה לתשלום\nמכירות וחשבוניות.", encoding="utf-8")
        (tmp_path / "cat" / "outlook.exe.md").write_text("# outlook reference", encoding="utf-8")

    def test_reads_subfolders_in_the_current_language(self, tmp_path):
        from kb import recall
        self._tree(tmp_path)
        content, name = recall("chrome.exe", kb_dir=tmp_path, query="my printer does not print", language="en")
        assert name == "the knowledge folder"
        assert "### en/README.md" in content and "### en/03-printer.md" in content
        assert "he/03-printer.md" not in content and "cat/he/O2C.md" not in content
        # the printer guide outranks the camera guide for a printer question
        assert content.index("03-printer.md") < content.index("01-camera.md")

    def test_hebrew_reads_hebrew_folders_and_unmarked_files(self, tmp_path):
        from kb import recall
        self._tree(tmp_path)
        content, _ = recall("chrome.exe", kb_dir=tmp_path, query="המדפסת לא מדפיסה", language="he")
        assert "### he/03-printer.md" in content and "### cat/he/O2C.md" in content
        assert "en/03-printer.md" not in content

    def test_app_file_found_in_a_subfolder_comes_first(self, tmp_path):
        from kb import recall
        self._tree(tmp_path)
        content, name = recall("OUTLOOK.EXE", kb_dir=tmp_path, query="printer", language="en")
        assert name == "outlook.exe"
        assert content.startswith("# outlook reference")
        assert "### en/03-printer.md" in content

    def test_budget_keeps_whole_files_only(self, tmp_path):
        from kb import recall
        self._tree(tmp_path)
        content, _ = recall("chrome.exe", kb_dir=tmp_path, query="printer", language="en", max_chars=120)
        assert len(content) <= 120
        for block in content.split("\n\n### ")[1:]:
            assert block.rstrip().endswith((".", "printer", "reference")), block  # no file cut mid-text

    def test_nothing_in_language_returns_empty(self, tmp_path):
        from kb import recall
        (tmp_path / "en").mkdir()
        (tmp_path / "en" / "a.md").write_text("english only", encoding="utf-8")
        assert recall("chrome.exe", kb_dir=tmp_path, language="he") == ("", "")
