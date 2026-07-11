"""
api.py — FastAPI wrapper around pipeline.py.

Endpoints:
  GET  /          — service info
  GET  /health    — health check
  POST /analyze   — analyze a legal document (plain text)
  POST /extract   — extract text from PDF or TXT file upload
"""

import io
import os
import sys

import pdfplumber
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

sys.path.insert(0, os.path.dirname(__file__))

from dotenv import load_dotenv
from pipeline import LegalLensPipeline

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '.env'))

app = FastAPI(
    title="LegalLens API",
    description="Explains Indian legal documents in plain language with grounded citations.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

pipeline = LegalLensPipeline()


# ── schemas ───────────────────────────────────────────────────────────────────

class AnalyzeRequest(BaseModel):
    document_text: str


class ClaimResponse(BaseModel):
    claim_text: str
    source_section: str
    full_citation: str
    source_quote: str
    grounding_confidence: float
    confidence_flag: str


class AnalyzeResponse(BaseModel):
    document_type: str | None
    out_of_scope: bool
    reason: str
    claims: list[ClaimResponse]
    low_confidence_warning: bool
    avg_grounding_confidence: float
    retrieved_chunks: list[dict]


class ExtractResponse(BaseModel):
    text: str
    chars: int
    source: str  # "pdf" | "txt"


# ── endpoints ─────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "service": "LegalLens API",
        "version": "1.0.0",
        "supported_document_types": [
            "cheque_bounce", "eviction_notice", "fir",
            "rental_agreement", "consumer_complaint",
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


@app.post("/extract", response_model=ExtractResponse)
async def extract(file: UploadFile = File(...)):
    """
    Extract plain text from an uploaded PDF or TXT file.
    Frontend calls this first, then passes the returned text to /analyze.
    """
    filename = file.filename or ""
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""

    if ext not in ("pdf", "txt"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF and TXT files are supported. For images, please copy the text manually."
        )

    contents = await file.read()

    if ext == "txt":
        try:
            text = contents.decode("utf-8")
        except UnicodeDecodeError:
            text = contents.decode("latin-1")
        return ExtractResponse(text=text.strip(), chars=len(text.strip()), source="txt")

    # PDF extraction via pdfplumber
    try:
        with pdfplumber.open(io.BytesIO(contents)) as pdf:
            pages = []
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    pages.append(page_text.strip())
            text = "\n\n".join(pages)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Could not extract text from PDF: {e}")

    if not text.strip():
        raise HTTPException(
            status_code=422,
            detail="No text could be extracted from this PDF. It may be a scanned image — please copy the text manually."
        )

    return ExtractResponse(text=text.strip(), chars=len(text.strip()), source="pdf")