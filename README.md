# LegalLens

An AI system that reads Indian legal documents and explains them in plain language — citing the exact statutory text behind every claim.

## What it does
- Accepts a legal document (currently: cheque bounce notices under NI Act)
- Retrieves relevant statutory sections from a curated legal corpus
- Generates a plain-language explanation where every claim is grounded in a retrieved source quote
- Scores each claim with a grounding confidence score to detect hallucination

## Eval results (cheque bounce, NI Act Ch. XVII)
| Metric | Score |
|---|---|
| Citation hit rate | 0.714 |
| Avg grounding confidence | 0.816 |
| Uncited claim rate | 0.000 |
| Out-of-scope accuracy | 1.000 |
