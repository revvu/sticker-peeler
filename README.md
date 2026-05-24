# Sticker Peeler

Static Three.js Rubik's Cube UI with bundled movement audio.

## Run

From this directory:

```sh
python3 -m http.server 8000
```

Open:

```text
http://127.0.0.1:8000/index.html
```

## Controls

- Drag to orbit the cube.
- Press `F`, `B`, `U`, `D`, `R`, or `L` to rotate a face clockwise.
- Hold `Shift` with a face key for counterclockwise rotation.
- Use `Scramble` for a random animated 20-move scramble.

## Audio

Movement audio is bundled at `assets/rubik-move.mp3`, from Pixabay's "Rubik Cube Small Noises" sound effect. The app plays short analyzed movement slices from the recording.
