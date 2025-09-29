# tests/conftest.py
import os
import sys
from pathlib import Path

# garante que a raiz do projeto esteja no sys.path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import json
import types
from fastapi.testclient import TestClient
from src import api as api_module

def _ensure_models_and_threshold():
    """
    Garante que exista um modelo/threshold para testes.
    Usa os mesmos caminhos esperados pela API.
    """
    models_dir = ROOT / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    # threshold padrão
    thr_file = models_dir / "decision_threshold.json"
    if not thr_file.exists():
        thr_file.write_text(json.dumps({"threshold": 0.59}, ensure_ascii=False), encoding="utf-8")

    
    model_path_cv = models_dir / "model_cv.joblib"
    model_path = models_dir / "model.joblib"
    if not model_path_cv.exists() and not model_path.exists():
        # cria um “modelo” fake com predict_proba
        class DummyModel:
            def predict_proba(self, X):
                import numpy as np
                # retorna probabilidade alta quando score_tecnico >= 0.3
                probs = []
                for row in X:
                    # X é texto composto; só para manter compatibilidade
                    probs.append([0.2, 0.8])
                return np.array(probs)

        import joblib
        joblib.dump(DummyModel(), models_dir / "model_cv.joblib")

_ensure_models_and_threshold()

# ===================== fixtures =====================

import pytest

@pytest.fixture(scope="session")
def api_app():
    # reimporta o app com os artefatos garantidos
    return api_module.app

@pytest.fixture(scope="session")
def api_client(api_app):
    return TestClient(api_app)

@pytest.fixture()
def api_payload():
    return {
        "job_text": "QA Selenium Java",
        "cand_text": "Selenium WebDriver, Java, testes automatizados",
        "score_tecnico": 0.25,
        "situacao_norm": "prospect",
    }
