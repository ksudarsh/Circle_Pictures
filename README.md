> **Archived 2026-10-16.** The code from this project (`process_images.py`, README, requirements) was copied into the Parakalamatham website repository (`ksudarsh/Parakalamatham_Website`) at `tools/circle-pictures/`; the circular portraits are kept outside Git. This repository keeps the full history and images, read-only.

# Circle Pictures

Interactive tool to frame PNG images in a circular, gold-rimmed frame and manually fill gaps using tiled samples.

## Requirements
- Conda (Miniconda/Anaconda) with Python 3.9+ available.
- Tkinter (ships with most Conda Python builds; if missing on Ubuntu, install `sudo apt-get install -y python3-tk`).

## Setup (Conda)
1) Create and activate the env
   ```bash
   conda create -n circle-pictures python=3.11 -y
   conda activate circle-pictures
   ```
2) Install dependencies
   ```bash
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```

## Run
1) Put your PNG files in `images/` (folder is auto-created if missing).
2) Run:
   ```
   python process_images.py
   ```
3) Outputs are written to `images_circular/`.

## Editor Controls
- Left-drag: draw sample tile selection; releasing captures the tile and shows a preview window.
- Left-click (or right-click): bucket-fill the clicked region (inside the circle) with the tiled sample.
- Hold Space + drag or use arrow keys: pan.
- `z`/`o`: zoom in/out.
- `u`: undo last fill.
- Close window or use the Save button to finish; Save writes the current view to the output image.

## Notes
- Keep source images as PNG to preserve transparency.
- If Tkinter is missing on Linux, install with `sudo apt-get install -y python3-tk`.
