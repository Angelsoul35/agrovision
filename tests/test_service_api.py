from __future__ import annotations

from fastapi.testclient import TestClient

from agrovision.generate import LABELS
from agrovision.main import app

client = TestClient(app)


def test_labels_endpoint():
    r = client.get("/labels")
    assert r.status_code == 200
    assert r.json()["labels"] == LABELS


def test_assess_sample_endpoint():
    r = client.post("/assess-sample", json={"label": "chlorosis", "seed": 3})
    assert r.status_code == 200
    body = r.json()
    assert body["true_label"] == "chlorosis"
    assert body["label"] in LABELS
    assert 0.0 <= body["confidence"] <= 1.0
    assert body["recommended_action"]


def test_assess_base64_roundtrip():
    sample = client.get("/sample/healthy", params={"seed": 4}).json()
    r = client.post("/assess", json={"image_base64": sample["image_base64"]})
    assert r.status_code == 200
    body = r.json()
    assert body["label"] == "healthy"
    assert body["method"] in ("rule", "model")


def test_assess_rejects_bad_image():
    r = client.post("/assess", json={"image_base64": "not-a-real-png!!!"})
    assert r.status_code == 400


def test_unknown_sample_label_is_404():
    r = client.get("/sample/banana")
    assert r.status_code == 404
