"""Acting on the user's behalf: action tags, step tags, and the code that runs them.

With ACTIONS_MODE on, the model may append action tags to its spoken answer,
in the same spirit as the [POINT] cursor tag and the draw-on-screen shapes:

    [CLICK:x,y]             click at screenshot pixel (x,y), POINT's coordinate space
    [TYPE:text]             type text into the focused control
    [KEYS:ctrl+shift+s]     press a key combination
    [POWERSHELL:command]    run a PowerShell command, output goes back to the model
    [OPEN:target]           open an app ("notepad", "chrome"), a file, or a URL

Guided steps (STEP_MODE) use two more tags, with no payload:

    [STEP]                  at least one more step follows this one
    [DONE]                  the task is finished

This module strips the tags (so TTS never reads them) and executes actions.
Clicks and keys go through pynput's controllers (already the hotkey
dependency); PowerShell runs as a child process with no window. app.py
decides WHEN to run them: at once, or after the user confirms with
Ctrl+Alt+Enter. A PowerShell command always waits for that confirmation.
"""
from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class Click:
    x: int
    y: int


@dataclass(frozen=True)
class Type:
    text: str


@dataclass(frozen=True)
class Keys:
    combo: str


@dataclass(frozen=True)
class PowerShell:
    command: str


@dataclass(frozen=True)
class Open:
    target: str


Action = Click | Type | Keys | PowerShell | Open

# Case-insensitive like the shape tags: the prompt asks for lowercase prose, so
# `[click:...]` must parse and strip too. TYPE and POWERSHELL payloads may hold
# a ']' (an array index, a regex), so they run to the ']' that is followed by
# another tag or the end of the text, not to the first one.
_CLICK_RE = re.compile(r"\[\s*CLICK\s*:\s*(\d+)\s*,\s*(\d+)\s*\]", re.IGNORECASE)
_KEYS_RE = re.compile(r"\[\s*KEYS\s*:\s*([^\]]+?)\s*\]", re.IGNORECASE)
_TYPE_RE = re.compile(r"\[\s*TYPE\s*:(.*?)\](?=\s*(?:\[|$))", re.IGNORECASE | re.DOTALL)
_POWERSHELL_RE = re.compile(r"\[\s*POWERSHELL\s*:(.*?)\](?=\s*(?:\[|$))", re.IGNORECASE | re.DOTALL)
_OPEN_RE = re.compile(r"\[\s*OPEN\s*:(.*?)\](?=\s*(?:\[|$))", re.IGNORECASE | re.DOTALL)
_ANY_ACTION_RE = re.compile(
    r"\[\s*(?:CLICK|KEYS)\s*:[^\]]*\]|\[\s*(?:TYPE|POWERSHELL|OPEN)\s*:.*?\](?=\s*(?:\[|$))",
    re.IGNORECASE | re.DOTALL,
)
# An action tag the stream cut off: strip it and everything after, like the
# shape parser does, so no half-tag is ever spoken.
_UNTERMINATED_ACTION_RE = re.compile(
    r"\[\s*(?:CLICK|KEYS|TYPE|POWERSHELL|OPEN)\s*:.*$", re.IGNORECASE | re.DOTALL
)
_STEP_RE = re.compile(r"\[\s*(STEP|DONE)\s*\]", re.IGNORECASE)

_PS_TIMEOUT_S = 60
_PS_OUTPUT_MAX = 2000


def parse_actions(text: str) -> tuple[str, list[Action]]:
    """Split the model's text into speech and the actions it asked for, in the
    order written. Everything that is not an action tag is returned as-is,
    so the [POINT] tag and the shape tags still parse afterwards."""
    found: list[tuple[int, Action]] = []
    for m in _CLICK_RE.finditer(text):
        found.append((m.start(), Click(int(m.group(1)), int(m.group(2)))))
    for m in _KEYS_RE.finditer(text):
        found.append((m.start(), Keys(m.group(1).strip().lower())))
    for m in _TYPE_RE.finditer(text):
        found.append((m.start(), Type(m.group(1))))
    for m in _POWERSHELL_RE.finditer(text):
        command = m.group(1).strip()
        if command:
            found.append((m.start(), PowerShell(command)))
    for m in _OPEN_RE.finditer(text):
        target = m.group(1).strip()
        if target:
            found.append((m.start(), Open(target)))
    found.sort(key=lambda item: item[0])
    stripped = _ANY_ACTION_RE.sub("", text)
    stripped = _UNTERMINATED_ACTION_RE.sub("", stripped)
    stripped = re.sub(r"[ \t]{2,}", " ", stripped).strip()
    return stripped, [action for _, action in found]


def parse_step_tag(text: str) -> tuple[str, str | None]:
    """Strip [STEP] / [DONE] and say which one was there ("step", "done" or
    None). With both, the last one written wins."""
    marks = [m.group(1).lower() for m in _STEP_RE.finditer(text)]
    stripped = re.sub(r"[ \t]{2,}", " ", _STEP_RE.sub("", text)).strip()
    return stripped, (marks[-1] if marks else None)


def needs_confirmation(actions: list[Action], mode: str) -> bool:
    """Whether app.py should hold these until Ctrl+Alt+Enter. Everything waits
    in 'confirm' mode; in 'auto' mode only a PowerShell command does."""
    if not actions:
        return False
    if mode == "auto":
        return any(isinstance(a, PowerShell) for a in actions)
    return True


def describe(action: Action) -> str:
    """One line for the debug log and for the note sent back to the model."""
    if isinstance(action, Click):
        return f"click at ({action.x}, {action.y})"
    if isinstance(action, Type):
        return f"type {action.text!r}"
    if isinstance(action, Keys):
        return f"press {action.combo}"
    if isinstance(action, Open):
        return f"open {action.target}"
    return f"powershell: {action.command}"


# --- Execution ---------------------------------------------------------------

_KEY_NAMES = {
    "ctrl": "ctrl", "control": "ctrl", "alt": "alt", "shift": "shift",
    "win": "cmd", "windows": "cmd", "super": "cmd",
    "enter": "enter", "return": "enter", "tab": "tab", "esc": "esc", "escape": "esc",
    "space": "space", "backspace": "backspace", "delete": "delete", "del": "delete",
    "home": "home", "end": "end", "pageup": "page_up", "pagedown": "page_down",
    "up": "up", "down": "down", "left": "left", "right": "right", "insert": "insert",
}


def _pynput_key(name: str, keyboard):
    """A pynput key for one name out of a combo. Letters and digits are
    themselves; the rest are looked up on keyboard.Key."""
    name = name.strip().lower()
    if len(name) == 1:
        return name
    if re.fullmatch(r"f([1-9]|1[0-2])", name):
        return getattr(keyboard.Key, name)
    mapped = _KEY_NAMES.get(name)
    if mapped is None:
        raise ValueError(f"unknown key {name!r}")
    return getattr(keyboard.Key, mapped)


def press_keys(combo: str, keyboard=None) -> None:
    """Press a combination like "ctrl+shift+s": modifiers held, last key tapped."""
    if keyboard is None:
        from pynput import keyboard
    names = [n for n in combo.replace(" ", "").split("+") if n]
    if not names:
        raise ValueError("empty key combination")
    keys = [_pynput_key(n, keyboard) for n in names]
    controller = keyboard.Controller()
    for key in keys[:-1]:
        controller.press(key)
    try:
        controller.press(keys[-1])
        controller.release(keys[-1])
    finally:
        for key in reversed(keys[:-1]):
            controller.release(key)


def click_at(x: int, y: int, mouse=None) -> None:
    """A left click at physical virtual-desktop pixel (x, y)."""
    if mouse is None:
        from pynput import mouse
    controller = mouse.Controller()
    controller.position = (x, y)
    controller.click(mouse.Button.left, 1)


def type_text(text: str, keyboard=None) -> None:
    """Type text into whatever has focus. pynput sends characters the current
    layout lacks (Hebrew on an English layout, say) as Unicode input."""
    if keyboard is None:
        from pynput import keyboard
    keyboard.Controller().type(text)


def run_powershell(command: str, runner=subprocess.run) -> str:
    """Run one PowerShell command without a window and return what it printed,
    cut to a size the model can read. A non-zero exit or a timeout is reported
    in the text rather than raised, so the model hears about it next turn."""
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    try:
        proc = runner(
            ["powershell", "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass",
             "-Command", command],
            capture_output=True, text=True, timeout=_PS_TIMEOUT_S, creationflags=flags,
            encoding="utf-8", errors="replace",
        )
    except subprocess.TimeoutExpired:
        return f"timed out after {_PS_TIMEOUT_S} seconds"
    except Exception as exc:  # powershell missing, permissions
        return f"could not run: {type(exc).__name__}: {exc}"
    out = (proc.stdout or "").strip()
    err = (proc.stderr or "").strip()
    parts = []
    if out:
        parts.append(out)
    if err:
        parts.append(f"stderr: {err}")
    if proc.returncode:
        parts.append(f"exit code {proc.returncode}")
    text = "\n".join(parts) or "(no output)"
    if len(text) > _PS_OUTPUT_MAX:
        text = text[:_PS_OUTPUT_MAX] + "\n... (cut)"
    return text


def open_target(target: str, startfile=None, popen=None) -> None:
    """Open an app, a file or a web address the way the Start menu or a
    double-click would. os.startfile handles paths and URLs (the default
    browser); an app name such as "notepad" or "chrome" goes through
    `start`, which also looks it up in the registry's App Paths."""
    import os
    startfile = startfile or os.startfile
    popen = popen or subprocess.Popen
    try:
        startfile(target)
        return
    except OSError:
        pass
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    proc = popen(["cmd", "/c", "start", "", target], creationflags=flags)
    try:
        if proc.wait(timeout=5) != 0:
            raise RuntimeError(f"could not open {target!r}")
    except subprocess.TimeoutExpired:
        pass  # start returned the console but the app is still launching


def execute(
    actions: list[Action],
    to_physical: Callable[[int, int], tuple[int, int]],
    log: Callable[[str], None] = lambda _msg: None,
    *,
    clicker=None,
    typer=None,
    presser=None,
    shell=None,
    opener=None,
) -> list[str]:
    """Run actions in order and return one result line per action. A failing
    action is reported and the rest still run; the model gets the lines.
    The doers default to this module's functions, looked up at call time so
    a test can patch them."""
    clicker = clicker or click_at
    typer = typer or type_text
    presser = presser or press_keys
    shell = shell or run_powershell
    opener = opener or open_target
    results: list[str] = []
    for action in actions:
        what = describe(action)
        try:
            if isinstance(action, Click):
                px, py = to_physical(action.x, action.y)
                clicker(px, py)
                results.append(f"{what}: done")
            elif isinstance(action, Type):
                typer(action.text)
                results.append(f"{what}: done")
            elif isinstance(action, Keys):
                presser(action.combo)
                results.append(f"{what}: done")
            elif isinstance(action, Open):
                opener(action.target)
                results.append(f"{what}: done")
            else:
                output = shell(action.command)
                results.append(f"{what}\noutput:\n{output}")
        except Exception as exc:
            results.append(f"{what}: failed ({type(exc).__name__}: {exc})")
        log(f"ACTION: {results[-1].splitlines()[0]}")
    return results
