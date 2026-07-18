# PPE Compliance Monitor

A local web app that detects Personal Protective Equipment (PPE) usage from images, videos, or a browser camera snapshot, and scores compliance over time — not just per frame.

## Tech Stack

- **Backend:** FastAPI (Python) + Uvicorn
- **Frontend:** Jinja2 templates, vanilla HTML/CSS/JS
- **Detection model:** YOLO (Ultralytics), custom-trained weights (`best.pt`)
- **CV/Numerics:** OpenCV, NumPy, PyTorch (CUDA if available, else CPU)

## Detected Classes

`Person`, `Glasses`, `Gloves`, `Helmet`, `Mask`, `Vest`, `Shoes`

## Core Algorithms

- **YOLO object detection** — locates people and PPE items in each frame.
- **Custom IOU + appearance-based tracker** (`SimpleBoxTracker`) — matches detections across frames using box IOU, position/size prediction, and an appearance feature score, so the same worker/item is tracked instead of re-detected from scratch each frame.
- **Worker identity resolution** (`WorkerIdentityResolver`) — re-links a worker's identity across short gaps (occlusion, missed frames) using track history.
- **Spatial PPE-to-person assignment** (`spatial.py`) — assigns each detected PPE item to the correct person using body-region priors (e.g. helmet near the top, shoes near the bottom) combined with overlap and horizontal-distance scoring.
- **Temporal compliance scoring** (`temporal.py`), inspired by an STCA (Spatio-Temporal Compliance Analysis) approach:
  - **Adaptive Compliance Scoring (ACSF-style):** exponentially decayed weighted average over a sliding window, so recent frames matter more than older ones.
  - **Temporal Entropy (TEMF-style):** measures stability/uncertainty of compliance over the tracked window to smooth out flickering detections.
  - Produces a trend label (`improving` / `declining` / `stable` / `warming-up`) and a final compliant/violation/at-risk status per worker.
- **Presence smoothing** (`PPEPresenceSmoother`) — persists an item's "present" state for a few frames to avoid flicker from momentary missed detections.

## Features

- Image, video, and browser-camera analysis from a single local UI
- Configurable required PPE set, confidence threshold, compliance threshold, and temporal window
- Annotated output images/videos saved to `runs/`
- JSON compliance reports and CSV event logs (per-worker violation streaks) for video runs

## Project Structure

```
app.py                  FastAPI entrypoint and API routes
ppe_app/config.py       Paths, defaults, app constants
ppe_app/tracking.py     YOLO wrapper, box tracker, worker identity resolver
ppe_app/spatial.py      PPE-to-person spatial assignment logic
ppe_app/temporal.py     Temporal compliance scoring (STCA-inspired)
ppe_app/engine.py       Orchestrates detection -> tracking -> scoring -> rendering
ppe_app/rendering.py    Frame/video annotation
ppe_app/reporting.py    Report and CSV event generation
ppe_app/helpers.py      Shared geometry/feature utilities
templates/index.html    Web UI
static/                 CSS and frontend JS
```

## Setup & Run

```bash
pip install -r requirements.txt
python app.py
```

Then open:

```
http://127.0.0.1:8000
```

The app uses `best.pt` by default if present in the project root; any other `.pt` file placed there will also show up as a selectable model.

## Notes

- `Shoes` is detected but not required by default in compliance checks, since site policy varies.
- If a browser can't preview a generated annotated video, use the download link in the results panel (depends on codecs available in your local OpenCV build).
