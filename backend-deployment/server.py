"""
Raqeeb — Compliance Confidence API (deployable, no LLM required)
This is the cloud-deployable version: the Confidence Score is pure math,
so it runs anywhere with zero GPU / Ollama. The RAG/LLM features stay local
(on-premise) — which is the product's whole value proposition.
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional

app = FastAPI(title="Raqeeb Compliance API")

# Allow the frontend (any origin) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Weights (must sum to 1.0) — Raqeeb's design decision ──
WEIGHTS = {
    "evidence_quality": 0.30,
    "coverage":         0.30,
    "freshness":        0.20,
    "audit_readiness":  0.20,
}


class Control(BaseModel):
    id: str
    status: str
    verified: bool = False
    evidence_quality: Optional[float] = None
    evidence_age_days: Optional[int] = None


class ConfidenceRequest(BaseModel):
    controls: List[Control]
    company_id: Optional[int] = 1


def score_evidence_quality(controls):
    scored = [c.evidence_quality for c in controls if c.evidence_quality is not None]
    return sum(scored) / len(scored) if scored else 0.0


def score_coverage(controls):
    with_ev = [c for c in controls if c.evidence_quality is not None]
    return len(with_ev) / len(controls) if controls else 0.0


def score_freshness(controls):
    fresh = []
    for c in controls:
        if c.evidence_age_days is None:
            continue
        age = c.evidence_age_days
        if age <= 90:
            fresh.append(1.0)
        elif age <= 180:
            fresh.append(0.6)
        elif age <= 365:
            fresh.append(0.3)
        else:
            fresh.append(0.1)
    return sum(fresh) / len(fresh) if fresh else 0.0


def score_audit_readiness(controls):
    ready = [c for c in controls
             if c.evidence_quality is not None
             and c.verified
             and c.status in ("compliant", "complete")]
    return len(ready) / len(controls) if controls else 0.0


def compute_confidence(controls):
    components = {
        "evidence_quality": score_evidence_quality(controls),
        "coverage":         score_coverage(controls),
        "freshness":        score_freshness(controls),
        "audit_readiness":  score_audit_readiness(controls),
    }
    total = sum(components[k] * WEIGHTS[k] for k in WEIGHTS)
    return round(total * 100), {k: round(v * 100) for k, v in components.items()}


@app.get("/")
def health():
    return {"status": "ok", "service": "Raqeeb Compliance API"}


@app.post("/confidence")
def confidence(req: ConfidenceRequest):
    score, breakdown = compute_confidence(req.controls)
    return {"score": score, "breakdown": breakdown}
