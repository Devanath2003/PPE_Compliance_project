# Anakha 2.0 PPE Compliance Monitor

This project turns your trained `best.pt` model into a full local product: a FastAPI web app with image analysis, video analysis, camera snapshot support, annotated outputs, and STCA-inspired temporal reasoning.

## What it implements

- YOLOv11 inference using the trained project weights in this folder
- STCA-style temporal compliance scoring inspired by the paper
  - TEMF-style temporal entropy tracking
  - ACSF-style adaptive compliance scoring
- Generalized PPE logic for the model's 7 classes
  - `Person`
  - `Glasses`
  - `Gloves`
  - `Helmet`
  - `Mask`
  - `Vest`
  - `Shoes`
- A local web UI for image, video, and browser camera snapshot analysis
- Output artifacts saved under `runs/`
  - annotated images or videos
  - JSON reports
  - CSV event logs for video runs

## Project structure

- `app.py`: FastAPI entrypoint
- `ppe_app/config.py`: paths and runtime defaults
- `ppe_app/temporal.py`: temporal scoring utilities
- `ppe_app/engine.py`: model loading, frame analysis, rendering, video processing
- `templates/index.html`: main UI
- `static/styles.css`: dashboard styling
- `static/app.js`: frontend logic
- `stca.py`: original/reference STCA module kept for context
- `inf.ipynb`: original prototype notebook kept for context

## How to run

1. Open a terminal in this project folder.
2. Start the app:

```bash
python app.py
```

3. Open:

```text
http://127.0.0.1:8000
```

## Notes

- The app defaults to `best.pt` if it exists.
- The paper focused on helmet and vest, but the UI lets you choose the required PPE set to match your site rules.
- `Shoes` is available in the UI but is not forced by default because site policy can differ.
- Annotated video encoding depends on codecs available through the local OpenCV build. If the browser does not preview the generated video, use the download links from the results panel.

## Reference basis

The implementation was built from:

- `Advancing_Industrial_Safety_A_Spatio-Temporal_Framework_for_PPE_Detection_Using_YOLOv11.pdf`
- `best.pt`
- `stca.py`
- `inf.ipynb`

