# MarketLens Deployment Guide

## Readiness

MarketLens is ready for a private Streamlit Community Cloud course or portfolio deployment. It is not an enterprise production system yet: the optional access code is a lightweight gate, not user authentication; provider spending limits, durable report storage, formal audit logs, and a source-quality policy would still be required for a customer-facing service.

## Before You Deploy

1. Confirm `.env` and `.streamlit/secrets.toml` are ignored by Git.
2. Create the `market-lens-sources` Pinecone index once with `python scripts/setup_pinecone.py`.
3. Run `pytest -q` locally.
4. Keep the deployment private or set a strong `APP_ACCESS_CODE`. The live workflow can consume Tavily and model credits.

## Streamlit Community Cloud

1. Push this repository to GitHub.
2. Open [Streamlit Community Cloud](https://share.streamlit.io/) and choose **Create app**.
3. Select the `nishokvg/market-lens-arra` repository, the `main` branch, and `app.py` as the entrypoint.
4. Open **Advanced settings** and paste the contents of `.streamlit/secrets.toml.example`, replacing every empty value with your real key. Do not upload or commit your `.env` file.
5. Deploy. Keep the app private while testing; public deployment without the access code can expose paid API usage.

## Smoke Test

1. Confirm the app accepts the access code.
2. Run one demo-mode request with three competitors.
3. Switch to live mode and run one request with three competitors before using all ten.
4. Confirm the status line shows `Mem0`, `Pinecone retrieval`, `Pinecone indexing`, and `LLM extraction` as expected.
5. Confirm every non-`Unknown` comparison value has a cited URL.
6. Approve and download a JSON report.

## Rollback

If provider errors, unexpected spend, or incorrect live claims appear, set `DEMO_MODE = "true"` in Streamlit Secrets and reboot the app from the Community Cloud management page. This disables live provider calls while retaining a safe demonstration workflow.
