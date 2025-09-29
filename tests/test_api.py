import json

def test_health_ok(api_client):
    resp = api_client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert set(data.keys()) == {"status", "model_loaded", "threshold"}
    assert data["status"] == "ok"
    assert isinstance(data["model_loaded"], bool)
    # threshold deve ser número (float)
    assert isinstance(data["threshold"], (int, float))


def test_predict_ok(api_client, api_payload):
    resp = api_client.post("/predict", content=json.dumps(api_payload))
    assert resp.status_code == 200

    data = resp.json()
    # estrutura básica
    assert "y_prob" in data and "y_pred" in data and "details" in data
    assert isinstance(data["y_prob"], (int, float))
    assert 0.0 <= data["y_prob"] <= 1.0
    assert data["y_pred"] in (0, 1)

    details = data["details"]
    # detalhes retornados
    for k in ("mode", "score_tecnico", "situacao_norm", "threshold"):
        assert k in details

    # consistência entre probabilidade e threshold
    thr = float(details["threshold"])
    y_prob = float(data["y_prob"])
    y_pred = int(data["y_pred"])
    assert (y_pred == 1 and y_prob >= thr) or (y_pred == 0 and y_prob < thr)


def test_predict_validation_error_missing_job_text(api_client):
    
    bad_payload = {
        "cand_text": "texto do candidato",
        "score_tecnico": 0.1,
        "situacao_norm": "prospect",
    }
    resp = api_client.post("/predict", content=json.dumps(bad_payload))
    assert resp.status_code == 422
