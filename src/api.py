# src/api.py
from __future__ import annotations

import json
import time
import logging
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, Request
from pydantic import BaseModel, Field

# --------------------------------------------------------------------------------------
# Configs e caminhos
# --------------------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
DATA_DIR = ROOT / "data"
THRESHOLD_FILE = MODELS_DIR / "decision_threshold.json"

DEFAULT_THRESHOLD = 0.59

_model = None
_threshold = DEFAULT_THRESHOLD
_model_loaded = False


def _load_threshold() -> float:
    if THRESHOLD_FILE.exists():
        try:
            data = json.loads(THRESHOLD_FILE.read_text(encoding="utf-8"))
            return float(data.get("threshold", DEFAULT_THRESHOLD))
        except Exception:
            return DEFAULT_THRESHOLD
    return DEFAULT_THRESHOLD


def _load_model():
    """Tenta carregar model_cv.joblib e, se não existir, model.joblib."""
    global _model, _model_loaded
    for path in (MODELS_DIR / "model_cv.joblib", MODELS_DIR / "model.joblib"):
        if path.exists():
            _model = joblib.load(path)
            _model_loaded = True
            return
    raise RuntimeError(
        "Nenhum modelo encontrado. Treine com: python src\\train_baseline.py ou python src\\train_cv.py"
    )


_threshold = _load_threshold()
try:
    _load_model()
except Exception:
    _model_loaded = False

# --------------------------------------------------------------------------------------
# Schemas
# --------------------------------------------------------------------------------------
class PredictRequest(BaseModel):
    job_text: str = Field(..., min_length=1)
    cand_text: str = Field(..., min_length=1)
    score_tecnico: float = Field(..., ge=0.0)
    situacao_norm: str = Field(..., min_length=1)
    mode: Optional[str] = Field(default="raw")


class PredictResponse(BaseModel):
    y_prob: float
    y_pred: int
    details: dict


# --------------------------------------------------------------------------------------
# App
# --------------------------------------------------------------------------------------
app = FastAPI(
    title="Decision Match API",
    version="1.0.0",
    description="API para predição de match candidato-vaga (baseline TF-IDF + features simples).",
)

# --- Logging de requisições (latência + status) ---
logger = logging.getLogger("uvicorn.access")

@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    dur_ms = (time.perf_counter() - start) * 1000
    logger.info(f"{request.method} {request.url.path} {response.status_code} {dur_ms:.1f}ms")
    return response


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": bool(_model_loaded), "threshold": _threshold}


# --------------------------------------------------------------------------------------
# Helpers de predição (tolerantes a diferentes formatos do pipeline)
# --------------------------------------------------------------------------------------
def _predict_proba_flexible(model, req_dict: dict) -> float:
    """
    Tenta prever com múltiplos formatos de entrada:
    (1) DF com colunas originais do treino
    (2) DF com coluna 'text' concatenada
    (3) lista 1D com string concatenada
    """
    # (1) tentar com colunas originais
    try:
        df_full = pd.DataFrame([{
            "job_text": req_dict["job_text"],
            "cand_text": req_dict["cand_text"],
            "situacao_norm": req_dict["situacao_norm"],
            "score_tecnico": req_dict["score_tecnico"],
        }])
        return float(model.predict_proba(df_full)[0][1])
    except Exception:
        pass

    # monta texto concatenado uma vez só
    text_concat = (
        f"[JOB]{req_dict['job_text']} "
        f"[CAND]{req_dict['cand_text']} "
        f"[SIT]{req_dict['situacao_norm']} "
        f"[SCORE]{req_dict['score_tecnico']}"
    )

    # (2) tentar com DF {'text': ...}
    try:
        df_text = pd.DataFrame({"text": [text_concat]})
        return float(model.predict_proba(df_text)[0][1])
    except Exception:
        pass

    # (3) tentar com lista 1D
    try:
        return float(model.predict_proba([text_concat])[0][1])
    except Exception as e:
        # se nada deu certo, propaga o último erro (mais informativo)
        raise e


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if not _model_loaded:
        _load_model()

    req_dict = {
        "job_text": req.job_text,
        "cand_text": req.cand_text,
        "situacao_norm": req.situacao_norm,
        "score_tecnico": req.score_tecnico,
    }

    proba = _predict_proba_flexible(_model, req_dict)
    y_pred = int(proba >= _threshold)

    return PredictResponse(
        y_prob=proba,
        y_pred=y_pred,
        details={
            "mode": req.mode,
            "score_tecnico": req.score_tecnico,
            "situacao_norm": req.situacao_norm,
            "threshold": _threshold,
        },
    )
