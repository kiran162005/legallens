"""
pipeline.py — the single entry point for LegalLens.

Flow:
  document_text
      │
      ▼
  [scope check]  ← is this document type supported yet?
      │
      ▼
  [retrieve]     ← top-k statutory chunks from corpus
      │
      ▼
  [generate]     ← structured (claim, citation, source_quote) list
      │
      ▼
  [confidence]   ← per-claim grounding_confidence score + low_confidence_warning flag
      │
      ▼
  PipelineResult (structured dict, ready for API or eval)

Note on classification:
  With only one document type (cheque bounce) we do a simple keyword-based
  scope check rather than a full ML classifier. The classifier module gets
  added in the next sprint when eviction notices go in — at that point we
  have two corpora and something real to distinguish between.
"""

import os
import re
from dataclasses import dataclass, asdict
from difflib import SequenceMatcher

from retrieve import CorpusRetriever, RetrievedChunk
from generate import GroundedExplainer
from dotenv import load_dotenv
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '.env'))
# ── config ────────────────────────────────────────────────────────────────────

SUPPORTED_TYPES = {
    "cheque_bounce": {
        "corpus_path": os.path.join(
            os.path.dirname(__file__), "..", "corpus", "ni_act_cheque_bounce.json"
        ),
        "keywords": [
            "cheque", "dishonour", "dishonored", "bounced", "insufficient funds",
            "negotiable instruments", "section 138", "demand notice", "unpaid",
            "drawer", "payee", "bank memo", "returned unpaid",
        ],
        "description": "Cheque bounce demand notice (NI Act Section 138)",
    },

    "eviction_notice": {
        "corpus_path": os.path.join(
            os.path.dirname(__file__), "..", "corpus", "karnataka_rent_act_eviction.json"
        ),
        "keywords": [
            "vacate", "eviction", "evict", "notice to quit", "tenant", "landlord",
            "premises", "possession", "rent arrears", "leave the premises",
            "rental agreement", "notice to vacate", "recovery of possession",
            "karnataka rent", "terminate tenancy",
        ],
        "description": "Eviction notice (Karnataka Rent Act 1999)",
    },

    "fir": {
    "corpus_path": os.path.join(
        os.path.dirname(__file__), "..", "corpus", "bnss_bns_fir.json"
    ),
    "keywords": [
        "fir", "first information report", "police complaint", "police station",
        "cognizable", "theft", "stolen", "assault", "cheating", "fraud",
        "deceived", "hurt", "attack", "accused", "complainant", "investigation",
        "section 173", "bnss", "bns",
    ],
    "description": "First Information Report (BNSS + BNS — theft, assault, cheating)",
},

    "rental_agreement": {
    "corpus_path": os.path.join(
        os.path.dirname(__file__), "..", "corpus", "rental_agreement_acts.json"
    ),
    "keywords": [
        "rental agreement", "rent agreement", "lease agreement", "lease deed",
        "stamp duty", "security deposit", "notice period", "monthly rent",
        "landlord", "tenant", "lessor", "lessee", "licence fee",
        "e-stamp", "registration", "11 month", "lock-in period",
    ],
    "description": "Rental/lease agreement (Registration Act + Karnataka Stamp Act + Contract Act)",
},
    # eviction_notice, rental_agreement, fir, consumer_complaint → added in next sprint
}

TOP_K_CHUNKS = 6          # retrieve top 4 chunks per query
CONFIDENCE_THRESHOLD = 0.35  # below this → low_confidence_warning on that claim


# ── scope detection ───────────────────────────────────────────────────────────

def detect_document_type(text: str) -> str | None:
    """
    Returns the matched document type key or None if no supported type matches.
    Simple keyword overlap for now; replaced by ML classifier once 2+ types exist.
    """
    text_lower = text.lower()
    best_type, best_score = None, 0

    for doc_type, config in SUPPORTED_TYPES.items():
        score = sum(1 for kw in config["keywords"] if kw in text_lower)
        if score > best_score:
            best_score, best_type = score, doc_type

    # require at least 2 keyword hits to avoid false positives
    return best_type if best_score >= 2 else None


# ── grounding confidence ──────────────────────────────────────────────────────

def compute_grounding_confidence(claim_text: str, source_quote: str, chunk_text: str) -> float:
    """
    Measures how well a generated claim is grounded in its cited source.

    Two signals combined:
    1. Quote presence — is the source_quote actually a substring (or near-match)
       of the retrieved chunk text? This catches the most basic hallucination:
       citing a real section but inventing the quote.
    2. Claim-quote similarity — does the claim say something semantically close
       to the quote, or is it a non-sequitur citation?

    Returns a float in [0, 1]. Below CONFIDENCE_THRESHOLD → flag as low confidence.
    """
    chunk_lower = chunk_text.lower()
    quote_lower = source_quote.lower().strip()

    # Signal 1: is the quote present in the chunk? (exact or near-match via SequenceMatcher)
    if quote_lower in chunk_lower:
        quote_presence = 1.0
    else:
        # fuzzy: longest common substring ratio
        quote_presence = SequenceMatcher(None, quote_lower, chunk_lower).ratio()
        quote_presence = min(quote_presence * 1.5, 1.0)  # scale up slightly, cap at 1

    # Signal 2: word overlap between claim and quote (proxy for semantic relevance)
    claim_words = set(re.findall(r'\w+', claim_text.lower()))
    quote_words = set(re.findall(r'\w+', quote_lower))
    stop_words = {"the", "a", "an", "is", "are", "was", "were", "of", "in",
                  "to", "and", "or", "for", "that", "this", "it", "with", "be"}
    claim_words -= stop_words
    quote_words -= stop_words
    if quote_words:
        overlap = len(claim_words & quote_words) / len(quote_words)
    else:
        overlap = 0.0

    # weighted combination: quote presence matters more
    confidence = 0.65 * quote_presence + 0.35 * overlap
    return round(confidence, 3)


def find_chunk_by_citation(citation: str, retrieved_chunks: list[RetrievedChunk]) -> RetrievedChunk | None:
    """Find the retrieved chunk whose citation matches the one the model cited."""
    citation_lower = citation.lower().strip()

    # first try: match both section number AND act name keywords
    act_keywords = {
        "registration act": "registration act",
        "stamp act": "stamp act",
        "contract act": "contract act",
        "transfer of property": "transfer of property",
        "negotiable instruments": "negotiable instruments",
        "karnataka rent": "karnataka rent",
        "bnss": "bnss",
        "bns": "bns",
    }
    citation_act = None
    for key in act_keywords:
        if key in citation_lower:
            citation_act = key
            break

    for chunk in retrieved_chunks:
        chunk_citation_lower = chunk.full_citation.lower()
        if chunk.section_number.lower() in citation_lower:
            if citation_act is None or citation_act in chunk_citation_lower:
                return chunk

    # fallback: section number only
    for chunk in retrieved_chunks:
        if chunk.section_number.lower() in citation_lower:
            return chunk

    return None

# ── main pipeline ─────────────────────────────────────────────────────────────

@dataclass
class AnnotatedClaim:
    claim_text: str
    source_section: str
    full_citation: str
    source_quote: str
    grounding_confidence: float
    confidence_flag: str  # "ok" | "low" | "uncited"


@dataclass
class PipelineResult:
    document_type: str | None
    out_of_scope: bool
    reason: str
    claims: list[AnnotatedClaim]
    low_confidence_warning: bool
    avg_grounding_confidence: float
    retrieved_chunks: list[dict]  # for transparency / frontend display


class LegalLensPipeline:
    def __init__(self, groq_api_key: str | None = None):
        self._retrievers: dict[str, CorpusRetriever] = {}
        self.explainer = GroundedExplainer(api_key=groq_api_key)

    def _get_retriever(self, doc_type: str) -> CorpusRetriever:
        if doc_type not in self._retrievers:
            corpus_path = SUPPORTED_TYPES[doc_type]["corpus_path"]
            self._retrievers[doc_type] = CorpusRetriever(corpus_path=corpus_path)
        return self._retrievers[doc_type]

    def run(self, document_text: str) -> dict:
        # ── 1. scope check ────────────────────────────────────────────────────
        doc_type = detect_document_type(document_text)
        if doc_type is None:
            return asdict(PipelineResult(
                document_type=None,
                out_of_scope=True,
                reason=(
                    "This document does not appear to match any supported document type. "
                    "Currently supported: cheque bounce notices (NI Act Section 138). "
                    "Eviction notices, FIRs, rental agreements, and consumer complaints "
                    "will be supported in a future version."
                ),
                claims=[],
                low_confidence_warning=False,
                avg_grounding_confidence=0.0,
                retrieved_chunks=[],
            ))

        # ── 2. retrieve ───────────────────────────────────────────────────────
        retriever = self._get_retriever(doc_type)

        # multi-query retrieval: search on full doc + specific sub-queries
        # this handles short documents that don't surface all relevant sections
        sub_queries_map = {
    "cheque_bounce": [
        document_text,
        "criminal complaint court filing magistrate cognizance",
        "settlement compounding payment dispute resolution",
        "trial procedure summary conviction punishment",
    ],

    "eviction_notice": [
        document_text,
        "grounds for eviction landlord tenant recovery possession",
        "notice period rent arrears non-payment Karnataka",
        "bona fide personal requirement subletting misuse premises",
    ],

    "fir": [
    document_text,
    "FIR registration police duty cognizable offence procedure",
    "theft stolen property punishment dwelling house",
    "assault hurt grievous hurt criminal force punishment",
    "cheating fraud deception property wrongful loss",
    ],

    "rental_agreement": [
    document_text,
    "registration compulsory lease one year stamp duty",
    "unstamped agreement inadmissible evidence penalty",
    "lessor lessee rights liabilities rent maintenance",
    "free consent essential elements valid contract",
],
}
        sub_queries = sub_queries_map.get(doc_type, [document_text])

        seen_ids = set()
        retrieved = []
        for query in sub_queries:
            for chunk in retriever.search(query, top_k=3):
                if chunk.chunk_id not in seen_ids:
                    seen_ids.add(chunk.chunk_id)
                    retrieved.append(chunk)
        retrieved = retrieved[:TOP_K_CHUNKS + 3]  # cap at 9 to avoid flooding context

        # ── 3. generate ───────────────────────────────────────────────────────
        raw_output = self.explainer.explain(document_text, retrieved)

        if raw_output.get("out_of_scope"):
            return asdict(PipelineResult(
                document_type=doc_type,
                out_of_scope=True,
                reason=raw_output.get("reason_if_out_of_scope", "Model flagged document as out of scope."),
                claims=[],
                low_confidence_warning=False,
                avg_grounding_confidence=0.0,
                retrieved_chunks=[{"chunk_id": c.chunk_id, "citation": c.full_citation, "text": c.text} for c in retrieved],
            ))

        # ── 4. annotate grounding confidence per claim ────────────────────────
        annotated_claims = []
        for raw_claim in raw_output.get("claims", []):
            matched_chunk = find_chunk_by_citation(
                raw_claim.get("full_citation", ""), retrieved
            )
            if matched_chunk is None:
                # model cited a section that wasn't in the retrieved set → hallucination risk
                annotated_claims.append(AnnotatedClaim(
                    claim_text=raw_claim.get("claim_text", ""),
                    source_section=raw_claim.get("source_section", ""),
                    full_citation=raw_claim.get("full_citation", ""),
                    source_quote=raw_claim.get("source_quote", ""),
                    grounding_confidence=0.0,
                    confidence_flag="uncited",
                ))
            else:
                conf = compute_grounding_confidence(
                    claim_text=raw_claim.get("claim_text", ""),
                    source_quote=raw_claim.get("source_quote", ""),
                    chunk_text=matched_chunk.text,
                )
                annotated_claims.append(AnnotatedClaim(
                    claim_text=raw_claim.get("claim_text", ""),
                    source_section=raw_claim.get("source_section", ""),
                    full_citation=raw_claim.get("full_citation", ""),
                    source_quote=raw_claim.get("source_quote", ""),
                    grounding_confidence=conf,
                    confidence_flag="ok" if conf >= CONFIDENCE_THRESHOLD else "low",
                ))

        # ── 5. aggregate ──────────────────────────────────────────────────────
        if annotated_claims:
            avg_conf = round(
                sum(c.grounding_confidence for c in annotated_claims) / len(annotated_claims), 3
            )
            low_conf_warning = avg_conf < CONFIDENCE_THRESHOLD or any(
                c.confidence_flag in ("low", "uncited") for c in annotated_claims
            )
        else:
            avg_conf = 0.0
            low_conf_warning = True

        return asdict(PipelineResult(
            document_type=doc_type,
            out_of_scope=False,
            reason="",
            claims=annotated_claims,
            low_confidence_warning=low_conf_warning,
            avg_grounding_confidence=avg_conf,
            retrieved_chunks=[
                {"chunk_id": c.chunk_id, "citation": c.full_citation, "score": c.score, "text": c.text}
                for c in retrieved
            ],
        ))


if __name__ == "__main__":
    import json

    sample_doc = """LEGAL NOTICE

    To: Mr. Rajesh Kumar, No. 45, 4th Cross, Jayanagar, Bangalore - 560011

    Sir, under instructions from my client Mr. Suresh Patel, I hereby inform you that
    Cheque No. 002145 dated 12-03-2026 for Rs. 3,50,000/- drawn on HDFC Bank was
    returned unpaid with the remark 'Funds Insufficient' on 18-03-2026.

    You are called upon to pay the said amount within 15 days of receipt of this notice,
    failing which criminal proceedings will be initiated under Section 138 of the
    Negotiable Instruments Act, 1881."""

    pipeline = LegalLensPipeline()  # reads GROQ_API_KEY from environment
    result = pipeline.run(sample_doc)
    print(json.dumps(result, indent=2))
