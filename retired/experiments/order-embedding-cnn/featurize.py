from __future__ import annotations

FACE_ORDER = ["F", "B", "U", "D", "R", "L"]
FACE_TO_INDEX = {face: index for index, face in enumerate(FACE_ORDER)}
NUM_FACE_TYPES = len(FACE_ORDER)
STICKER_COUNT = 54
FACE_GRID_SHAPE = (len(FACE_ORDER), 3, 3)


def state_key_to_indices(state_key: str) -> list[int]:
    if len(state_key) != STICKER_COUNT:
        raise ValueError(f"Expected state key length {STICKER_COUNT}, got {len(state_key)}")
    return [FACE_TO_INDEX[face] for face in state_key]


def indices_to_state_key(indices: list[int]) -> str:
    if len(indices) != STICKER_COUNT:
        raise ValueError(f"Expected indices length {STICKER_COUNT}, got {len(indices)}")
    return "".join(FACE_ORDER[index] for index in indices)


def state_key_to_face_grid(state_key: str) -> list[list[list[int]]]:
    indices = state_key_to_indices(state_key)
    grid = []
    offset = 0

    for _face in FACE_ORDER:
        face_rows = []
        for _row in range(3):
            face_rows.append(indices[offset : offset + 3])
            offset += 3
        grid.append(face_rows)

    return grid


def face_grid_to_indices(face_grid: list[list[list[int]]]) -> list[int]:
    if len(face_grid) != FACE_GRID_SHAPE[0]:
        raise ValueError(f"Expected {FACE_GRID_SHAPE[0]} faces, got {len(face_grid)}")

    indices = []
    for face in face_grid:
        if len(face) != FACE_GRID_SHAPE[1]:
            raise ValueError(f"Expected {FACE_GRID_SHAPE[1]} rows per face")
        for row in face:
            if len(row) != FACE_GRID_SHAPE[2]:
                raise ValueError(f"Expected {FACE_GRID_SHAPE[2]} columns per row")
            indices.extend(row)

    if len(indices) != STICKER_COUNT:
        raise ValueError(f"Expected {STICKER_COUNT} sticker indices, got {len(indices)}")

    return indices
