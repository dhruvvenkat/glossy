import json
import os
import re
import tempfile
import uuid
from datetime import datetime
from pathlib import Path

THREADS_DIR = Path("~/.config/glossy/threads").expanduser()
THREAD_RECENT_TURNS = 6
THREAD_SUMMARY_MAX_CHARS = 2500
THREAD_NAME_MAX_CHARS = 80
THREAD_PICKER_ROWS = 5
THREAD_PICKER_REQUESTED = object()


class ThreadStore:
    def __init__(self, root=THREADS_DIR):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(self.root, 0o700)
        self.state_path = self.root / "state.json"
        state = self._read(self.state_path, {"active_id": None, "enabled": False})
        self.active_id = state.get("active_id")
        self.enabled = bool(state.get("enabled"))

    def _read(self, path, default=None):
        if not path.exists() and default is not None:
            return default
        try:
            value = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as error:
            raise RuntimeError(f"Cannot load {path}: {error}") from error
        if not isinstance(value, dict):
            raise RuntimeError(f"{path} must contain a JSON object")
        return value

    def _write(self, path, value):
        descriptor, temporary = tempfile.mkstemp(
            dir=self.root, prefix=f".{path.name}.", text=True
        )
        temporary = Path(temporary)
        try:
            with os.fdopen(descriptor, "w") as output:
                json.dump(value, output, indent=2)
                output.write("\n")
                output.flush()
                os.fsync(output.fileno())
            os.chmod(temporary, 0o600)
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)

    def _save_state(self):
        self._write(
            self.state_path,
            {"active_id": self.active_id, "enabled": self.enabled},
        )

    def _path(self, thread_id):
        return self.root / f"thread-{thread_id}.json"

    def load(self, thread_id):
        thread = self._read(self._path(thread_id))
        required = {"id", "name", "summary", "summarized_turns", "turns"}
        if not required <= thread.keys() or not isinstance(thread["turns"], list):
            raise RuntimeError(f"Invalid thread file: {self._path(thread_id)}")
        return thread

    def list(self):
        threads = [self._read(path) for path in self.root.glob("thread-*.json")]
        return sorted(threads, key=lambda thread: thread["name"].casefold())

    def current(self):
        return self.load(self.active_id) if self.enabled and self.active_id else None

    def selected(self):
        return self.load(self.active_id) if self.active_id else None

    def create(self, name):
        name = " ".join(name.split())
        if not name:
            name = datetime.now().strftime("Thread %Y-%m-%d %H-%M-%S")
        if len(name) > THREAD_NAME_MAX_CHARS:
            raise ValueError(
                f"Thread names must be at most {THREAD_NAME_MAX_CHARS} characters."
            )
        if any(thread["name"].casefold() == name.casefold() for thread in self.list()):
            raise ValueError(f"Thread {name} already exists.")
        now = datetime.now().astimezone().isoformat(timespec="seconds")
        thread = {
            "id": uuid.uuid4().hex,
            "name": name,
            "created_at": now,
            "updated_at": now,
            "summary": "",
            "summarized_turns": 0,
            "turns": [],
        }
        self._write(self._path(thread["id"]), thread)
        self.active_id = thread["id"]
        self.enabled = True
        self._save_state()
        return thread

    def activate(self, name):
        match = next(
            (
                thread
                for thread in self.list()
                if thread["name"].casefold() == name.strip().casefold()
            ),
            None,
        )
        if match is None:
            raise ValueError(f"Thread {name.strip()} does not exist.")
        self.active_id = match["id"]
        self.enabled = True
        self._save_state()
        return match

    def set_enabled(self, enabled):
        if enabled and not self.active_id:
            raise ValueError("No thread is selected. Say new thread followed by a name.")
        self.enabled = enabled
        self._save_state()

    def append_turn(self, question, answer):
        thread = self.current()
        if thread is None:
            return None
        thread["turns"].append(
            {
                "question": question,
                "answer": answer,
                "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            }
        )
        thread["updated_at"] = thread["turns"][-1]["created_at"]
        self._write(self._path(thread["id"]), thread)
        return thread

    def save_summary(self, thread, summary):
        thread["summary"] = summary[:THREAD_SUMMARY_MAX_CHARS]
        thread["summarized_turns"] = len(thread["turns"])
        self._write(self._path(thread["id"]), thread)


def handle_thread_command(transcript, store):
    command = transcript.strip().rstrip(".!?").strip()
    new_thread = re.fullmatch(r"new thread(?:\s+(.+))?", command, re.IGNORECASE)
    switch_thread = re.fullmatch(
        r"switch to thread\s+(.+)", command, re.IGNORECASE
    )
    try:
        if new_thread:
            thread = store.create(new_thread.group(1) or "")
            return f"Started thread {thread['name']}."
        if switch_thread:
            thread = store.activate(switch_thread.group(1))
            return f"Switched to thread {thread['name']}."
        if command.casefold() == "list threads":
            threads = store.list()
            return (
                THREAD_PICKER_REQUESTED
                if threads
                else "You do not have any threads yet."
            )
        if command.casefold() == "current thread":
            thread = store.selected()
            if thread is None:
                return "No thread is selected."
            status = "on" if store.enabled else "off"
            return f"The current thread is {thread['name']}. Threads mode is {status}."
        if command.casefold() == "threads mode":
            store.set_enabled(True)
            return f"Threads mode is on for {store.selected()['name']}."
        if command.casefold() == "exit threads mode":
            store.set_enabled(False)
            return "Threads mode is off."
    except ValueError as error:
        return str(error)
    return None


def thread_picker_text(items, selected):
    start = max(
        0,
        min(selected - THREAD_PICKER_ROWS // 2, len(items) - THREAD_PICKER_ROWS),
    )
    end = min(len(items), start + THREAD_PICKER_ROWS)
    lines = ["Choose a thread", ""]
    if start:
        lines.append("  ↑ more")
    lines.extend(
        f"{'›' if index == selected else ' '} {items[index]['name']}"
        for index in range(start, end)
    )
    if end < len(items):
        lines.append("  ↓ more")
    lines.extend(["", "↑/↓ move   Enter select   Esc cancel"])
    return "\n".join(lines)


def thread_input(thread, question):
    parts = [f"Reading thread: {thread['name']}"]
    if thread["summary"]:
        parts.append(f"Thread memory:\n{thread['summary']}")
    if thread["turns"]:
        recent = []
        for turn in thread["turns"][-THREAD_RECENT_TURNS:]:
            recent.extend(
                [f"Reader: {turn['question']}", f"Assistant: {turn['answer']}"]
            )
        parts.append("Recent conversation:\n" + "\n".join(recent))
    parts.append(f"Current question:\n{question}")
    return "\n\n".join(parts)
