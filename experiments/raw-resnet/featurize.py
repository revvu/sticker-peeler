from __future__ import annotations

import torch

FACE_ORDER = ["F", "B", "U", "D", "R", "L"]
FACE_TO_INDEX = {face: index for index, face in enumerate(FACE_ORDER)}
NUM_COLORS = len(FACE_ORDER)
STICKER_COUNT = 54
IMAGE_HEIGHT = 6
IMAGE_WIDTH = 9
GRID_SIZE = 3

# Two rows of three 3x3 faces: [F, B, U] on top, [D, R, L] on bottom.
TILE_LAYOUT = [
    [0, 1, 2],
    [3, 4, 5],
]


def state_key_to_face_grids(state_key: str) -> list[list[list[int]]]:
    if len(state_key) != STICKER_COUNT:
        raise ValueError(f"Expected state key length {STICKER_COUNT}, got {len(state_key)}")

    grids: list[list[list[int]]] = []
    offset = 0

    for _face in FACE_ORDER:
        face_rows = []
        for _row in range(GRID_SIZE):
            row = [FACE_TO_INDEX[state_key[offset + column]] for column in range(GRID_SIZE)]
            face_rows.append(row)
            offset += GRID_SIZE
        grids.append(face_rows)

    return grids


def face_grids_to_index_image(face_grids: list[list[list[int]]]) -> list[list[int]]:
    image = [[0 for _ in range(IMAGE_WIDTH)] for _ in range(IMAGE_HEIGHT)]

    for tile_row, face_indices in enumerate(TILE_LAYOUT):
        for tile_col, face_index in enumerate(face_indices):
            face = face_grids[face_index]
            row_offset = tile_row * GRID_SIZE
            col_offset = tile_col * GRID_SIZE

            for row in range(GRID_SIZE):
                for column in range(GRID_SIZE):
                    image[row_offset + row][col_offset + column] = face[row][column]

    return image


def index_image_to_one_hot(index_image: list[list[int]]) -> torch.Tensor:
    tensor = torch.zeros(NUM_COLORS, IMAGE_HEIGHT, IMAGE_WIDTH, dtype=torch.float32)

    for row in range(IMAGE_HEIGHT):
        for column in range(IMAGE_WIDTH):
            color = index_image[row][column]
            tensor[color, row, column] = 1.0

    return tensor


def state_key_to_one_hot_image(state_key: str) -> torch.Tensor:
    face_grids = state_key_to_face_grids(state_key)
    index_image = face_grids_to_index_image(face_grids)
    return index_image_to_one_hot(index_image)
