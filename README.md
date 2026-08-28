# MarketLens AI for Arra Global

MarketLens AI is a human-approved apparel competitor-intelligence agent. It turns a sourcing brief into a cited comparison of competitors, then pauses for a human decision before a report can be saved.

## What the MVP does

- Accepts an apparel sourcing brief: categories, target market, requirements, and optional known competitors.
- Runs a LangGraph workflow for discovery, research, fact extraction, comparison, evidence review, and approval.
- Keeps claims separate from evidence and marks unsupported fields as `Unknown` rather than inventing them.
- Uses Tavily to discover and deeply research 3 to 10 competitors, collecting current public evidence when `TAVILY_API_KEY` is configured. It otherwise uses clearly labelled demonstration data so the full workflow can be shown safely.
- Recalls relevant researcher preferences with Mem0 when `MEM0_API_KEY` is configured, then writes only an approved report summary back to memory.
- Reuses and indexes source snippets through LlamaIndex and Pinecone when the Pinecone and OpenAI variables are configured.
- Supports an opt-in OpenAI structured extractor. It is constrained to return `Unknown` for unsupported fields and to attach an evidence quote and URL to each supported claim.
- Prepares a JSON download only after the user approves it in the Streamlit app; this works on Streamlit Community Cloud without relying on its temporary filesystem.
- Shows live stage progress in the interface while LangGraph completes the research workflow.

## Architecture

```text
Streamlit UI
    -> LangGraph workflow
       -> discovery tool -> research tool -> fact extraction -> comparison -> reviewer
                                                           -> human approval -> report export
```

Keep `USE_LLM_EXTRACTION=false` while rehearsing. Turn it on only after adding `OPENAI_API_KEY` and selecting a model in `.env`. Native OpenAI works with the default blank `OPENAI_BASE_URL`. For an OpenRouter key, set `OPENAI_BASE_URL=https://openrouter.ai/api/v1`, use provider-qualified model names (for example `openai/gpt-4.1-mini`), and set `OPENAI_EMBEDDING_MODEL=openai/text-embedding-3-small`.

## Run locally

```bash
cd /Users/nishokmini/project/market-lens-arra
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
cp .env.example .env
streamlit run app.py
```

Start with `DEMO_MODE=true`. To research current websites, set `TAVILY_API_KEY` and switch demo mode off.

If Pinecone is configured, create its required index once before starting the app:

```bash
python scripts/setup_pinecone.py
```

## Deploy safely

This project is ready for a Streamlit Community Cloud **portfolio or course deployment**, not an unrestricted public production service. It can use paid provider credits, so set `APP_ACCESS_CODE` in Streamlit Secrets before enabling live mode. Do not commit `.env` or `.streamlit/secrets.toml`.

See [the deployment guide](docs/DEPLOYMENT.md) for the exact GitHub, Secrets, and smoke-test steps.

## Project success criteria

- Three to ten competitors are compared in one report.
- Every non-unknown comparison field is backed by a URL and quoted evidence.
- The evidence coverage target is at least 80% of requested fields.
- A human approves the draft before it is persisted.
