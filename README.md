# neo-char
A dynamic 2D character overlay for streaming and videos.

> Currently only basic PNGtuber functionality is supported.

## Setup
1. Clone the repo `git clone https://github.com/AAK-27/neo-char.git`
2. Create a Python 3.12 virtual environment inside the repo `py -3.12 -m venv .venv`
3. Activate the venv and install dependencies

   `.\\.venv\\Scripts\\activate`

   `pip install -r requirements.txt`

4. Move your character sprites into the `images` folder
5. Update the sprite paths for each expression in `basic_pngtuber.py`
6. Run the script `python basic_pngtuber.py`
7. In OBS Studio, add a virtual camera source with a chroma key filter to display the overlay

## Notes
- Startup logs from MediaPipe (e.g. XNNPACK and feedback manager warnings) are expected and can be ignored
- Tune `SILENCE_THRESHOLD`, `BLINKING_THRESHOLD`, and `SCARED_THRESHOLD` in `basic_pngtuber.py` to match your setup
