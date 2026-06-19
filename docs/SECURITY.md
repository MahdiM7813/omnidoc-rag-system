# Security Notes

The input notebooks contained hardcoded secret-handling patterns. This repository intentionally uses environment variables and `.env.example` only.

Recommended practices:

1. Rotate any keys that were ever exposed in notebooks.
2. Store real secrets in a secret manager or deployment environment variables.
3. Never commit `.env`.
4. Treat uploaded PDFs as untrusted input.
5. Keep `allow_dangerous_deserialization: false` unless loading your own FAISS index.
6. Add API authentication before exposing the FastAPI service publicly.
