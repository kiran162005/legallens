# LegalLens

> AI system that reads Indian legal documents and explains them in plain language, citing the exact statutory text behind every claim, with measurable hallucination prevention.

---

## What it does

Upload or paste a legal document and LegalLens:

1. **Classifies** the document type automatically
2. **Retrieves** relevant statutory sections from a curated legal corpus using ChromaDB
3. **Generates** a plain-language explanation where every claim cites a specific section and quotes the exact statutory text it is grounded in
4. **Scores** each claim with a grounding confidence score, flagging weak or uncited claims

The core guarantee: the system only cites sections that were actually retrieved. It cannot hallucinate a law it was not given.

---

## Supported document types

| Document Type | Acts Covered |
|---|---|
| Cheque Bounce Notice | Negotiable Instruments Act 1881 (Ch. XVII, S.138-148) |
| Eviction Notice | Karnataka Rent Act 1999 |
| First Information Report | BNSS 2023 (S.173-176) + BNS 2023 (theft, assault, cheating) |
| Rental / Lease Agreement | Registration Act 1908, Karnataka Stamp Act 1957, Indian Contract Act 1872, Transfer of Property Act 1882 |
| Consumer Complaint | Consumer Protection Act 2019 |

---

## Eval results

Evaluated across 22 documents, 5 document types, 7 acts, 171 generated claims.

| Metric | Score |
|---|---|
| **Uncited claim rate** | **0.000** |
| **Out-of-scope accuracy** | **1.000** |
| Citation hit rate | 0.660 |
| Avg grounding confidence | 0.678 |

**Uncited claim rate = 0** means the model never cited a statutory section not in the retrieved context across all 171 claims.

### Real PDF test results

| Document | Claims | Avg Grounding | Uncited |
|---|---|---|---|
| Cheque Bounce Notice | 5 | 65% | 0 |
| Eviction Notice | 6 | 74% | 0 |
| FIR Complaint | 9 | 79% | 0 |
| Rental Agreement | 9 | 43% | 0 |
| Consumer Complaint | 9 | 49% | 0 |

### Citation hit rate improvement

| Change made | Citation hit rate |
|---|---|
| Baseline (single-query, top_k=4) | 0.286 |
| top_k 4 to 6 | 0.429 |
| Multi-query retrieval | 0.714 |
| ChromaDB metadata-filtered retrieval | 0.660 |

---

## Architecture

```
PDF/TXT upload or paste
        |
        v
[pdfplumber extraction]     PDF/TXT to plain text
        |
        v
[keyword classifier]        detect document type (5 types, min 3 keyword hits)
        |
        v
[multi-query retrieval]     4 sub-queries x top-3 ChromaDB search
                            metadata filter: only search relevant doc type chunks
        |
        v
[grounded generation]       Groq/Llama 3.1 with structured JSON schema
                            every claim must have source_section, full_citation, source_quote
        |
        v
[grounding check]           quote presence + word overlap scoring
                            act-name aware citation matching
                            flags: ok / low / uncited
        |
        v
PipelineResult              per-claim confidence scores + aggregate metrics
```

### Key design decisions

**Section-level chunking** - each corpus chunk is one statutory section or sub-section. Ensures every citation is a complete, citable unit.

**Structured generation schema** - the LLM is forced to output claim_text, source_section, full_citation, source_quote for every claim. Cannot make a free-form assertion without committing to a verifiable quote.

**Multi-query retrieval** - short documents do not surface all relevant sections in a single query. 4 domain-specific sub-queries improved citation hit rate from 0.429 to 0.714.

**ChromaDB with metadata filtering** - migrated from FAISS for persistent storage. Each query searches only the 9-12 chunks belonging to the detected document type, not all 55.

**Act-name citation matching** - when multiple acts share a section number, the matcher checks both section number and act name to avoid grounding against the wrong chunk.

---

## Stack

- **Vector DB**: ChromaDB (persistent, metadata-filtered)
- **Embeddings**: sentence-transformers/all-MiniLM-L6-v2
- **Generation**: Groq API, Llama 3.1 8B Instant
- **PDF extraction**: pdfplumber
- **Backend**: FastAPI + Uvicorn
- **Frontend**: Vanilla HTML/CSS/JS

---

## Project structure

```
legallens/
|-- corpus/
|   |-- ni_act_cheque_bounce.json         9 chunks, NI Act Ch. XVII
|   |-- karnataka_rent_act_eviction.json  12 chunks, KRA 1999
|   |-- bnss_bns_fir.json                 12 chunks, BNSS + BNS 2023
|   |-- rental_agreement_acts.json        11 chunks, 4 acts
|   +-- consumer_protection_act.json      11 chunks, CPA 2019
|-- backend/
|   |-- retrieve.py     ChromaDB retrieval with metadata filtering
|   |-- generate.py     Groq structured generation
|   |-- pipeline.py     classify -> retrieve -> generate -> score
|   +-- api.py          FastAPI wrapper
|-- eval/
|   |-- *_gold.json     hand-labeled gold sets (22 documents)
|   +-- results/        timestamped eval runs
|-- eval_run.py         eval harness
|-- frontend/
|   +-- index.html      single-page UI with PDF upload
+-- requirements.txt
```

---

## Run locally

```bash
git clone https://github.com/kiran162005/legallens.git
cd legallens
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env
# add your GROQ_API_KEY to .env

cd backend
uvicorn api:app --reload --port 8000
# open frontend/index.html in browser

cd ..
python eval_run.py
```

Note: On first startup ChromaDB embeds all 55 corpus chunks and saves to disk (~30 seconds). Subsequent startups load instantly.

---

## Corpus sources

- **India Code** (indiacode.nic.in) - NI Act, Registration Act, Transfer of Property Act, Indian Contract Act, BNS, BNSS
- **PRS India** (prsindia.org) - Karnataka Rent Act 1999
- **NCDRC** (ncdrc.nic.in) - Consumer Protection Act 2019
- **Karnataka Gazette** - Karnataka Stamp Act 1957

Each chunk records source_url and last_verified_date for provenance.

---

## Built by

Kiran T, CSE, Garden City University Bangalore (2023-2027)

[GitHub](https://github.com/kiran162005) | [LegalLens Repo](https://github.com/kiran162005/legallens)
