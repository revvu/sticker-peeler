from __future__ import annotations

from featurize import (
    FACE_ORDER,
    face_grid_to_indices,
    indices_to_state_key,
    state_key_to_face_grid,
    state_key_to_indices,
)

SOLVED_STATE_KEY = "".join(face * 9 for face in FACE_ORDER)


def test_round_trip() -> None:
    indices = state_key_to_indices(SOLVED_STATE_KEY)
    grid = state_key_to_face_grid(SOLVED_STATE_KEY)
    round_trip_indices = face_grid_to_indices(grid)
    round_trip_key = indices_to_state_key(round_trip_indices)

    assert indices == round_trip_indices
    assert round_trip_key == SOLVED_STATE_KEY
    assert len(grid) == 6
    assert all(len(face) == 3 and all(len(row) == 3 for row in face) for face in grid)


if __name__ == "__main__":
    test_round_trip()
    print("featurize round-trip ok")
