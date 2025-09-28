import pytest
from fastapi.testclient import TestClient
from src.api import app, PredictRequest

client = TestClient(app)


def test_health_ok():
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert "model_loaded" in body
    assert "threshold" in body


def test_predict_endpoint(monkeypatch):
    # mocka o modelo para garantir saída previsível
    class FakeModel:
        def predict_proba(self, X):
            return [[0.1, 0.9]]  # sempre prob da classe 1 = 0.9

    # força o _model e o _model_loaded dentro de src.api
    import src.api as api
    monkeypatch.setattr(api, "_model", FakeModel())
    monkeypatch.setattr(api, "_model_loaded", True)

    payload = {
        "job_text": "Desenvolvedor Python",
        "cand_text": "Experiência com Python e FastAPI",
        "situacao_norm": "prospect",
        "score_tecnico": 0.8,
    }

    resp = client.post("/predict", json=payload)
    assert resp.status_code == 200
    body = resp.json()

    # garante que a resposta bate com o mock
    assert pytest.approx(body["y_prob"], rel=1e-6) == 0.9
    assert body["y_pred"] == 1  # 0.9 >= threshold padrão (0.59)
    assert "details" in body
    assert body["details"]["threshold"] == api._threshold
