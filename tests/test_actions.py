"""Unit tests for actions.py: the action and step tags, and running them.

The model appends [CLICK]/[TYPE]/[KEYS]/[POWERSHELL] tags (and a [STEP] or
[DONE] mark) to its spoken answer; the parser strips them so the voice never
reads them, and execute() runs them through injected clickers/typers.
"""
from __future__ import annotations

import subprocess
from unittest.mock import MagicMock

import pytest


class TestParseActions:
    def test_click_and_keys_in_order(self):
        from actions import parse_actions, Click, Keys
        spoken, acts = parse_actions("i'll open it and save. [CLICK:120,40][KEYS:ctrl+s] [POINT:120,40:file]")
        assert spoken == "i'll open it and save. [POINT:120,40:file]"
        assert acts == [Click(120, 40), Keys("ctrl+s")]

    def test_type_keeps_brackets_inside_text(self):
        from actions import parse_actions, Type
        spoken, acts = parse_actions('typing it now. [TYPE:items[0] = "a"] [POINT:none]')
        assert acts == [Type('items[0] = "a"')]
        assert spoken == "typing it now. [POINT:none]"

    def test_powershell_command_with_brackets(self):
        from actions import parse_actions, PowerShell
        spoken, acts = parse_actions("listing the folder. [POWERSHELL:Get-ChildItem C:\\x | Select -First 3][STEP]")
        assert acts == [PowerShell("Get-ChildItem C:\\x | Select -First 3")]
        assert spoken == "listing the folder. [STEP]"

    def test_lowercase_tags_parse_and_strip(self):
        from actions import parse_actions, Click
        spoken, acts = parse_actions("ok. [click: 5 , 6 ]")
        assert acts == [Click(5, 6)]
        assert spoken == "ok."

    def test_unterminated_tag_is_cut_not_spoken(self):
        from actions import parse_actions
        spoken, acts = parse_actions("let me type it [TYPE:hello wor")
        assert spoken == "let me type it"
        assert acts == []

    def test_plain_text_untouched(self):
        from actions import parse_actions
        text = "the save button is up top [POINT:30,40:save]"
        assert parse_actions(text) == (text, [])

    def test_order_follows_position_not_kind(self):
        from actions import parse_actions, Click, Type, Keys
        _, acts = parse_actions("[TYPE:a][CLICK:1,2][KEYS:enter]")
        assert acts == [Type("a"), Click(1, 2), Keys("enter")]


class TestParseStepTag:
    def test_step(self):
        from actions import parse_step_tag
        assert parse_step_tag("click file. [STEP] [POINT:1,2:file]") == ("click file. [POINT:1,2:file]", "step")

    def test_done(self):
        from actions import parse_step_tag
        assert parse_step_tag("that's it, saved. [done]") == ("that's it, saved.", "done")

    def test_none(self):
        from actions import parse_step_tag
        assert parse_step_tag("html is markup.") == ("html is markup.", None)


class TestNeedsConfirmation:
    def test_confirm_mode_holds_everything(self):
        from actions import needs_confirmation, Click
        assert needs_confirmation([Click(1, 1)], "confirm") is True

    def test_auto_mode_runs_clicks_but_holds_powershell(self):
        from actions import needs_confirmation, Click, PowerShell
        assert needs_confirmation([Click(1, 1)], "auto") is False
        assert needs_confirmation([Click(1, 1), PowerShell("dir")], "auto") is True

    def test_nothing_needs_nothing(self):
        from actions import needs_confirmation
        assert needs_confirmation([], "confirm") is False


class TestExecute:
    def test_runs_each_kind_and_maps_click_to_physical(self):
        from actions import execute, Click, Type, Keys, PowerShell
        clicker, typer, presser, shell = MagicMock(), MagicMock(), MagicMock(), MagicMock(return_value="out")
        results = execute(
            [Click(10, 20), Type("hi"), Keys("ctrl+s"), PowerShell("dir")],
            to_physical=lambda x, y: (x * 2, y * 2),
            clicker=clicker, typer=typer, presser=presser, shell=shell,
        )
        clicker.assert_called_once_with(20, 40)
        typer.assert_called_once_with("hi")
        presser.assert_called_once_with("ctrl+s")
        shell.assert_called_once_with("dir")
        assert results[0] == "click at (10, 20): done"
        assert results[3] == "powershell: dir\noutput:\nout"

    def test_failure_reported_and_rest_still_run(self):
        from actions import execute, Click, Keys
        presser = MagicMock()
        results = execute(
            [Click(1, 1), Keys("enter")],
            to_physical=lambda x, y: (_ for _ in ()).throw(RuntimeError("no screen")),
            clicker=MagicMock(), presser=presser,
        )
        assert "failed (RuntimeError: no screen)" in results[0]
        presser.assert_called_once_with("enter")


class TestPressKeys:
    def test_modifiers_held_last_key_tapped(self):
        from actions import press_keys
        kb = MagicMock()
        kb.Key.ctrl = "CTRL"; kb.Key.shift = "SHIFT"
        ctl = kb.Controller.return_value
        press_keys("ctrl+shift+s", keyboard=kb)
        assert [c.args[0] for c in ctl.press.call_args_list] == ["CTRL", "SHIFT", "s"]
        assert [c.args[0] for c in ctl.release.call_args_list] == ["s", "SHIFT", "CTRL"]

    def test_names_map_to_pynput_keys(self):
        from actions import press_keys
        kb = MagicMock()
        kb.Key.cmd = "WIN"; kb.Key.page_down = "PGDN"; kb.Key.f5 = "F5"
        ctl = kb.Controller.return_value
        press_keys("win+pagedown", keyboard=kb)
        press_keys("f5", keyboard=kb)
        assert [c.args[0] for c in ctl.press.call_args_list] == ["WIN", "PGDN", "F5"]

    def test_unknown_key_raises(self):
        from actions import press_keys
        with pytest.raises(ValueError):
            press_keys("ctrl+banana", keyboard=MagicMock())


class TestRunPowershell:
    def test_no_window_and_output_returned(self):
        from actions import run_powershell
        runner = MagicMock(return_value=MagicMock(stdout="hello\n", stderr="", returncode=0))
        assert run_powershell("echo hello", runner=runner) == "hello"
        args, kwargs = runner.call_args
        assert args[0][:4] == ["powershell", "-NoProfile", "-NonInteractive", "-ExecutionPolicy"]
        assert args[0][-1] == "echo hello"
        assert kwargs["creationflags"] == getattr(subprocess, "CREATE_NO_WINDOW", 0)

    def test_error_and_exit_code_in_text(self):
        from actions import run_powershell
        runner = MagicMock(return_value=MagicMock(stdout="", stderr="bad", returncode=1))
        assert run_powershell("x", runner=runner) == "stderr: bad\nexit code 1"

    def test_timeout_is_text_not_exception(self):
        from actions import run_powershell
        runner = MagicMock(side_effect=subprocess.TimeoutExpired("powershell", 60))
        assert "timed out" in run_powershell("sleep", runner=runner)

    def test_long_output_is_cut(self):
        from actions import run_powershell, _PS_OUTPUT_MAX
        runner = MagicMock(return_value=MagicMock(stdout="x" * 5000, stderr="", returncode=0))
        out = run_powershell("x", runner=runner)
        assert out.endswith("(cut)") and len(out) < _PS_OUTPUT_MAX + 20
