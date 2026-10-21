"""Shared pytest fixtures."""
import keyring
import pytest

# Tests must not see, or write, the developer's own saved settings and keys:
# config.py resolves them at import. Tests that need a stored value patch the
# module they test (settings_dialog.keyring, config.keyring) themselves.
keyring.get_password = lambda *_a, **_k: None
keyring.set_password = lambda *_a, **_k: None


@pytest.fixture(autouse=True)
def _debug_sessions_in_tmp(tmp_path, monkeypatch):
    """Per-interaction debug folders go to a temp dir, never the user's real one."""
    import debug_log

    monkeypatch.setattr(debug_log, "_DEBUG_DIR", tmp_path / "debug")
