# LegalLens

> AI system that reads Indian legal documents and explains them in plain language — citing the exact statutory text behind every claim, with measurable hallucination prevention.

![LegalLens Demo](docs/demo.png)

---

## What it does

Paste a legal document (cheque bounce notice, eviction notice, FIR, rental agreement, or consumer complaint) and LegalLens:

1. **Classifies** the document type
2. **Retrieves** the most relevant statutory sections from a curated legal corpus
3. **Generates** a plain-language explanation where every claim cites a specific section and quotes the exact statutory text it's grounded in
4. **Scores** each claim with a grounding confidence score — flagging weak or uncited claims before they mislead you

The core guarantee: the system only cites sections that were actually retrieved. It cannot hallucinate a law it wasn't given.

---

## Supported document types

| Document Type | Acts Covered |
|---|---|
| Cheque Bounce Notice | Negotiable Instruments Act 1881 (Ch. XVII, S.138–148) |
| Eviction Notice | Karnataka Rent Act 1999 |
| First Information Report | BNSS 2023 (S.173–176) + BNS 2023 (theft, assault, cheating) |
| Rental / Lease Agreement | Registration Act 1908 · Karnataka Stamp Act 1957 · Indian Contract Act 1872 · Transfer of Property Act 1882 |
| Consumer Complaint | Consumer Protection Act 2019 |

---

## Eval results

Evaluated across 22 documents, 5 document types, 7 acts, 152 generated claims.

| Metric | Score |
|---|---|
| **Uncited claim rate** | **0.000** |
| **Out-of-scope accuracy** | **1.000** |
| Citation hit rate | 0.574 |
| Avg grounding confidence | 0.725 |
| Low conf claim rate | 0.059 |

**Uncited claim rate = 0** means the model never cited a statutory section that wasn't in the retrieved context — across all 152 claims. This is the core anti-hallucination guarantee.

### Improvement trajectory (citation hit rate)

| Sprint | Change made | Citation hit rate |
|---|---|---|
| Baseline | Single-query retrieval, top_k=4 | 0.286 |
| Prompt tuning | Better system prompt | 0.286 |
| top_k increase | top_k 4 → 6 | 0.429 |
| Multi-query retrieval | 4 sub-queries per document type | **0.714** |

---

## Architecture

```
document_text
     │
     ▼
[keyword classifier]      detect document type (5 types)
     │
     ▼
[multi-query retrieval]   4 queries × top-3 FAISS dense search → deduplicated chunks
     │
     ▼
[grounded generation]     Groq/Llama 3.1 with structured JSON schema
                          every claim must have: source_section, full_citation, source_quote
     │
     ▼
[grounding check]         quote presence (SequenceMatcher) + word overlap
                          flags: ok / low / uncited
     │
     ▼
PipelineResult            structured output with per-claim confidence scores
```

### Key design decisions

**Section-level chunking** — each corpus chunk = one statutory section or sub-section. Prevents retrieval from returning half a section and ensures every citation is a complete, citable unit.

**Structured generation schema** — the LLM is forced to output `{claim_text, source_section, full_citation, source_quote}` for every claim. It cannot make a free-form assertion without committing to a specific quote that can be programmatically verified.

**Multi-query retrieval** — short documents don't surface all relevant sections in a single query. Running 4 domain-specific sub-queries and merging results improved citation hit rate from 0.429 to 0.714 on test documents.

**Act-name citation matching** — when multiple acts have the same section number (e.g. Section 17 in both Registration Act and Karnataka Stamp Act), matching by section number alone causes grounding scores to compute against the wrong chunk. The citation matcher checks both section number and act name.

---

## Stack

- **Retrieval**: FAISS (IndexFlatIP, cosine similarity) + `sentence-transformers/all-MiniLM-L6-v2`
- **Generation**: Groq API (Llama 3.1 8B Instant), `temperature=0.1`, JSON mode
- **Backend**: FastAPI + Uvicorn
- **Frontend**: Vanilla HTML/CSS/JS (no framework)
- **Eval**: Custom harness with timestamped JSON results

---

## Project structure

```
legallens/
├── corpus/                          # statutory text, section-level chunks
│   ├── ni_act_cheque_bounce.json    # 9 chunks — NI Act Ch. XVII
│   ├── karnataka_rent_act_eviction.json  # 12 chunks — KRA 1999
│   ├── bnss_bns_fir.json            # 12 chunks — BNSS + BNS 2023
│   ├── rental_agreement_acts.json   # 11 chunks — 4 acts
│   └── consumer_protection_act.json # 11 chunks — CPA 2019
├── backend/
│   ├── retrieve.py                  # FAISS dense retrieval
│   ├── generate.py                  # Groq structured generation
│   ├── pipeline.py                  # classify → retrieve → generate → score
│   └── api.py                       # FastAPI wrapper
├── eval/
│   ├── *_gold.json                  # hand-labeled gold sets (22 documents)
│   └── results/                     # timestamped eval runs
├── eval_run.py                      # eval harness
├── frontend/
│   └── index.html                   # single-page UI
└── requirements.txt
```

---

## Run locally

```bash
# clone and install
git clone https://github.com/kiran162005/legallens.git
cd legallens
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt

# add your Groq API key
cp .env.example .env
# edit .env and paste your GROQ_API_KEY

# start backend
cd backend
uvicorn api:app --reload --port 8000

# open frontend
# open frontend/index.html in your browser
# or: cd frontend && python -m http.server 3000

# run eval
cd ..
python eval_run.py
```

---

## Corpus sources

All statutory text sourced verbatim from official government repositories:

- **India Code** (indiacode.nic.in) — NI Act, Registration Act, Transfer of Property Act, Indian Contract Act, BNS, BNSS
- **PRS India** (prsindia.org) — Karnataka Rent Act 1999
- **NCDRC** (ncdrc.nic.in) — Consumer Protection Act 2019
- **Karnataka Gazette** — Karnataka Stamp Act 1957

Each corpus chunk records `source_url` and `last_verified_date` for provenance.

---

## Built by

Kiran T — CSE, Garden City University Bangalore (2023–2027)

[GitHub](https://github.com/kiran162005) · [LegalLens Repo](https://github.com/kiran162005/legallens)