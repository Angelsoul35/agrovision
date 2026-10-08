"""FastAPI surface over the leaf-health service."""

from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .generate import LABELS
from .service import LeafService, encode_png

app = FastAPI(title="AgroVision", version="1.0.0")
_service: LeafService | None = None


def service() -> LeafService:
    global _service
    if _service is None:
        _service = LeafService()
    return _service


class AssessRequest(BaseModel):
    image_base64: str = Field(min_length=1, description="RGB leaf image, base64-encoded PNG")


class SampleRequest(BaseModel):
    label: str = Field(description=f"one of {LABELS}")
    seed: int = 0


def _assessment_dict(a) -> dict:
    return {
        "label": a.label,
        "confidence": round(a.confidence, 4),
        "method": a.method,
        "needs_review": a.needs_review,
        "recommended_action": a.recommended_action,
        "top_features": a.top_features,
    }


@app.get("/labels")
def labels() -> dict:
    return {"labels": LABELS}


@app.post("/assess")
def assess(req: AssessRequest) -> dict:
    """Assess a leaf supplied as a base64-encoded PNG."""
    try:
        assessment = service().assess_base64(req.image_base64)
    except Exception as exc:  # noqa: BLE001 - surface any decode/shape error as 400
        raise HTTPException(status_code=400, detail=f"could not read image: {exc}") from exc
    return _assessment_dict(assessment)


@app.get("/sample/{label}")
def sample(label: str, seed: int = 0) -> dict:
    """Return a deterministic sample leaf of ``label`` as a base64 PNG."""
    try:
        s = service().sample(label, seed)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {
        "sample_id": s.sample_id,
        "true_label": s.true_label,
        "image_base64": encode_png(s.image),
    }


@app.post("/assess-sample")
def assess_sample(req: SampleRequest) -> dict:
    """Generate a sample leaf of a known class and assess it (true label echoed)."""
    try:
        s, assessment = service().assess_sample(req.label, req.seed)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    body = _assessment_dict(assessment)
    body["sample_id"] = s.sample_id
    body["true_label"] = s.true_label
    body["correct"] = assessment.label == s.true_label
    return body
