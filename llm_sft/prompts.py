"""Fixed instruction prompts for title/author SFT."""

TITLE_INSTRUCTION = (
    "Extract the bibliographic TITLE from the Tibetan segment below. "
    "Reply with JSON only using the key \"spans\". "
    "Each span must include \"text\", \"start\", and \"end\" (0-based character "
    "offsets relative to the segment text, inclusive at end). "
    "Every span text must be an exact substring of the input. "
    "If there is no title, reply: {\"spans\": []}."
)

AUTHOR_INSTRUCTION = (
    "Extract the bibliographic AUTHOR from the Tibetan segment below. "
    "Reply with JSON only using the key \"spans\". "
    "Each span must include \"text\", \"start\", and \"end\" (0-based character "
    "offsets relative to the segment text, inclusive at end). "
    "Every span text must be an exact substring of the input. "
    "If there is no author, reply: {\"spans\": []}."
)

INSTRUCTIONS = {
    "title": TITLE_INSTRUCTION,
    "author": AUTHOR_INSTRUCTION,
}
