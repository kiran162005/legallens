"""
api.py — thin FastAPI wrapper around pipeline.py.

Single endpoint: POST /analyze
  - accepts a legal document as plain text
  - returns structured claims with citations and grounding scores

Run with:
  uvicorn api:app --reload --port 8000
"""

import os
import sys

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

sys.path.insert(0, os.path.dirname(__file__))

from pipeline import LegalLensPipeline
from dotenv import load_dotenv

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '.env'))

app = FastAPI(
    title="LegalLens API",
    description="Explains Indian legal documents in plain language with grounded citations.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten in production
    allow_methods=["*"],
    allow_headers=["*"],
)

# initialise pipeline once at startup — loads embedding model and corpus indexes
pipeline = LegalLensPipeline()


# ── request / response schemas ────────────────────────────────────────────────

class AnalyzeRequest(BaseModel):
    document_text: str

    class Config:
        json_schema_extra = {
            "example": {
                "document_text": "LEGAL NOTICE\n\nTo: Mr. Rajesh Kumar...\n\nCheque No. 002145 was returned unpaid with remark Funds Insufficient..."
            }
        }


class ClaimResponse(BaseModel):
    claim_text: str
    source_section: str
    full_citation: str
    source_quote: str
    grounding_confidence: float
    confidence_flag: str  # "ok" | "low" | "uncited"


class AnalyzeResponse(BaseModel):
    document_type: str | None
    out_of_scope: bool
    reason: str
    claims: list[ClaimResponse]
    low_confidence_warning: bool
    avg_grounding_confidence: float
    retrieved_chunks: list[dict]


# ── endpoints ─────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "service": "LegalLens API",
        "version": "1.0.0",
        "supported_document_types": [
            "cheque_bounce",
            "eviction_notice",
            "fir",
            "rental_agreement",
            "consumer_complaint",
        ],
        "docs": "/docs",
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(request: AnalyzeRequest):
    if not request.document_text or not request.document_text.strip():
        raise HTTPException(status_code=400, detail="document_text cannot be empty")

    if len(request.document_text) > 10000:
        raise HTTPException(status_code=400, detail="document_text too long (max 10,000 characters)")

    result = pipeline.run(request.document_text)
    return result