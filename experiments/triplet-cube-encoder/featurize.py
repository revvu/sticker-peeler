from __future__ import annotations

FACE_ORDER = ["F", "B", "U", "D", "R", "L"]
FACE_TO_INDEX = {face: index for index, face in enumerate(FACE_ORDER)}
MOVE_TO_INDEX = {
    f"{face}{suffix}": index
    for index, (face, suffix) in enumerate(
        (face, suffix)
        for face in FACE_ORDER
        for suffix in ("", "'", "2")
    )
}
NUM_FACE_TYPES = len(FACE_ORDER)
NUM_MOVE_TYPES = len(MOVE_TO_INDEX)
STICKER_COUNT = 54
MAX_MOVE_LENGTH = 15
EMPTY_MOVE_TOKEN = NUM_MOVE_TYPES


def state_key_to_indices(state_key: str) -> list[int]:
    if len(state_key) != STICKER_COUNT:
        raise ValueError(f"Expected state key length {STICKER_COUNT}, got {len(state_key)}")
    return [FACE_TO_INDEX[face] for face in state_key]
