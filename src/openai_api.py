from pathlib import Path

from threads import thread_input

SYSTEM_PROMPT = (Path(__file__).resolve().parent.parent / "config" / "system-prompt.md").read_text().strip()
THREAD_SUMMARY_EVERY = 5
THREAD_PROMPT = """You are answering within an ongoing reading thread. Use the supplied
thread memory and recent conversation as background. Do not invent details that are not
present, and prioritize the user's current question if it conflicts with older context."""


def answer(client, settings, question, thread=None):
    request = {
        "model": settings["model"],
        "instructions": (
            f"{SYSTEM_PROMPT}\n\n{THREAD_PROMPT}" if thread else SYSTEM_PROMPT
        ),
        "input": thread_input(thread, question) if thread else question,
    }
    if settings["reasoning_effort"] is not None:
        request["reasoning"] = {"effort": settings["reasoning_effort"]}
    response = client.responses.create(**request).output_text.strip()
    if not response:
        raise RuntimeError("OpenAI returned an empty answer")
    return response


def summarize(client, settings, thread):
    start = thread["summarized_turns"]
    if len(thread["turns"]) - start < THREAD_SUMMARY_EVERY:
        return None
    pending = "\n".join(
        f"Reader: {turn['question']}\nAssistant: {turn['answer']}"
        for turn in thread["turns"][start:]
    )
    request = {
        "model": settings["model"],
        "instructions": (
            "Maintain a concise, factual memory for an ongoing reading thread. "
            "Preserve important concepts, the reader's demonstrated understanding, "
            "and unresolved questions. Do not guess. Return plain text under 2000 characters."
        ),
        "input": (
            f"Existing memory:\n{thread['summary'] or 'None'}"
            f"\n\nNew conversation:\n{pending}"
        ),
    }
    if settings["reasoning_effort"] is not None:
        request["reasoning"] = {"effort": settings["reasoning_effort"]}
    summary = client.responses.create(**request).output_text.strip()
    if not summary:
        raise RuntimeError("OpenAI returned an empty thread summary")
    return summary
