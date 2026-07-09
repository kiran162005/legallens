"""
generate.py — takes a legal document + retrieved statutory chunks, and produces
a plain-language explanation as a list of structured (claim, citation, source_quote)
objects. This structure is itself the core anti-hallucination mechanism: the model
cannot make a free-form claim without also committing to a specific section and a
specific quoted snippet that a downstream checker can verify against the retrieved
text (see eval/grounding_check in eval_run.py).

Uses Groq's Llama 3.1 for generation (fast + cheap, matches prior project stack).
"""

import json
import os

from groq import Groq

SYSTEM_PROMPT = """You are LegalLens, an assistant that explains Indian legal documents in plain language.
"If you cannot find a source_quote from the retrieved text for a claim, omit that claim entirely. Never leave source_section or full_citation blank. "

You will be given:
1. The text of a legal document (e.g. a cheque bounce notice)
2. A set of retrieved statutory sections (the ONLY source of legal truth you may cite)

Your job: produce a list of plain-language claims explaining what is happening and what it means,
where EVERY claim is grounded in one of the retrieved sections.

STRICT RULES:
- You may ONLY cite sections that appear in the retrieved context. Never invent a section number,
  never cite a section you were not given, even if you "know" it from training.
- Every claim must include a `source_quote` field: an exact, verbatim substring copied from the
  retrieved section text that supports the claim. Do not paraphrase the quote — copy it exactly.
"- If a claim cannot be grounded in the retrieved sections, do not make it. Every single claim MUST have a non-empty source_section, full_citation, and source_quote. Omit the claim entirely rather than leaving these fields blank." It is better to omit
  a claim than to state it ungrounded.
- If the document does not appear to match the type of law you were given context for, set
  "out_of_scope": true and return an empty claims list, with a brief reason.

"Generate a MAXIMUM of 8 claims total. Prioritize the most important legal points only. Do not repeat similar claims. "
"You MUST attempt to generate a claim from every retrieved section provided to you, "
"as long as it is relevant to the document. Do not anchor only on the most prominent section. "
"Respond with ONLY valid JSON, no markdown fences, no preamble, matching this schema:":
{
  "out_of_scope": false,
  "reason_if_out_of_scope": "",
  "claims": [
    {
      "claim_text": "plain language explanation of this point",
      "source_section": "138",
      "full_citation": "Section 138, Negotiable Instruments Act, 1881",
      "source_quote": "exact verbatim substring from the retrieved section text"
    }
  ]
}
"""


def build_user_prompt(document_text: str, retrieved_chunks: list) -> str:
    context_block = "\n\n".join(
        f"[{c.full_citation}]\n{c.text}" for c in retrieved_chunks
    )
    return f"""DOCUMENT TO EXPLAIN:
{document_text}

RETRIEVED STATUTORY CONTEXT (your only permitted source of legal claims):
{context_block}

Produce the structured JSON explanation now."""


class GroundedExplainer:
    def __init__(self, api_key: str | None = None, model: str = "llama-3.1-8b-instant"):
        self.client = Groq(api_key=api_key or os.environ.get("GROQ_API_KEY"))
        self.model = model

    def explain(self, document_text: str, retrieved_chunks: list) -> dict:
        user_prompt = build_user_prompt(document_text, retrieved_chunks)

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,  # low temperature: we want consistent, conservative grounding, not creativity
            response_format={"type": "json_object"},
        )

        raw = response.choices[0].message.content
        
        try:
            result = json.loads(raw)
            # retry once if claims list is empty — LLM occasionally returns empty on short docs
            if not result.get("claims") and not result.get("out_of_scope"):
                response2 = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0.1,
                    response_format={"type": "json_object"},
                )
                result = json.loads(response2.choices[0].message.content)
            return result
        except json.JSONDecodeError as e:
            return {
                "out_of_scope": False,
                "reason_if_out_of_scope": "",
                "claims": [],
                "_error": f"Failed to parse model output as JSON: {e}",
                "_raw_output": raw,
            }


if __name__ == "__main__":
    # smoke test placeholder — requires GROQ_API_KEY and retrieve.py's CorpusRetriever
    # to actually run end to end. See pipeline.py for the wired-up version.
    print("generate.py loaded OK. Run pipeline.py for an end-to-end test.")
