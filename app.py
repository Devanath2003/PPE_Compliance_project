from __future__ import annotations

import json
import shutil
import uuid
from pathlib import Path

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.concurrency import run_in_threadpool

from ppe_app.config import (
    APP_SUBTITLE,
    APP_TITLE,
    PAPER_HIGHLIGHTS,
    ROOT_DIR,
    RUNS_DIR,
    UPLOADS_DIR,
    default_model_name,
)
from ppe_app.engine import PPEComplianceEngine


UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
RUNS_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title=APP_TITLE, version="1.0.0")
app.mount("/static", StaticFiles(directory=str(ROOT_DIR / "static")), name="static")
app.mount("/runs", StaticFiles(directory=str(RUNS_DIR)), name="runs")
templates = Jinja2Templates(directory=str(ROOT_DIR / "templates"))
engine = PPEComplianceEngine()


def _parse_required_ppe(raw: str | None) -> list[str] | None:
    if not raw:
        return None
    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, list) else None
    except json.JSONDecodeError:
        return None


def _store_upload(file: UploadFile) -> Path:
    suffix = Path(file.filename or "upload.bin").suffix.lower()
    target = UPLOADS_DIR / f"{uuid.uuid4().hex}{suffix}"
    with target.open("wb") as handle:
        shutil.copyfileobj(file.file, handle)
    return target


@app.get("/", response_class=HTMLResponse)
async def home(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "app_title": APP_TITLE,
            "app_subtitle": APP_SUBTITLE,
            "default_model": default_model_name() or "best.pt",
            "paper_highlights": PAPER_HIGHLIGHTS,
        },
    )


@app.get("/api/config")
async def get_config(model_name: str | None = None) -> JSONResponse:
    try:
        payload = await run_in_threadpool(engine.get_model_metadata, model_name)
        return JSONResponse(payload)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/health")
async def healthcheck() -> JSONResponse:
    return JSONResponse({"status": "ok", "title": APP_TITLE})


@app.post("/api/analyze/image")
async def analyze_image(
    file: UploadFile = File(...),
    model_name: str = Form(...),
    required_ppe: str | None = Form(None),
    confidence_threshold: float = Form(...),
    compliance_threshold: float = Form(...),
    temporal_window: int = Form(...),
) -> JSONResponse:
    upload_path: Path | None = None
    try:
        upload_path = _store_upload(file)
        payload = await run_in_threadpool(
            engine.analyze_image,
            upload_path,
            model_name,
            _parse_required_ppe(required_ppe),
            confidence_threshold,
            compliance_threshold,
            temporal_window,
        )
        return JSONResponse(payload)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        if upload_path and upload_path.exists():
            upload_path.unlink(missing_ok=True)


@app.post("/api/analyze/video")
async def analyze_video(
    file: UploadFile = File(...),
    model_name: str = Form(...),
    required_ppe: str | None = Form(None),
    confidence_threshold: float = Form(...),
    compliance_threshold: float = Form(...),
    temporal_window: int = Form(...),
) -> JSONResponse:
    upload_path: Path | None = None
    try:
        upload_path = _store_upload(file)
        payload = await run_in_threadpool(
            engine.analyze_video,
            upload_path,
            model_name,
            _parse_required_ppe(required_ppe),
            confidence_threshold,
            compliance_threshold,
            temporal_window,
        )
        return JSONResponse(payload)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        if upload_path and upload_path.exists():
            upload_path.unlink(missing_ok=True)


@app.exception_handler(HTTPException)
async def http_exception_handler(_: Request, exc: HTTPException) -> JSONResponse:
    return JSONResponse(status_code=exc.status_code, content={"error": exc.detail})


@app.exception_handler(Exception)
async def generic_exception_handler(_: Request, exc: Exception) -> JSONResponse:
    return JSONResponse(status_code=500, content={"error": str(exc)})


if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=False)
