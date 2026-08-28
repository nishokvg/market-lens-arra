# Week 3 Project Documentation: MarketLens AI for Arra Global

## Project Overview

**Project track:** Code-heavy, using LangChain and LangGraph.

**Use case:** Project 3A - Market Research Agent: Competitor Analysis, adapted for Arra Global's apparel sourcing business.

**One-liner:** MarketLens AI helps apparel sourcing teams research competitors in a Streamlit web app, replacing four to six hours of manual web research. It discovers and researches competitors, extracts evidence-backed sourcing facts with four read tools, hands the report to a human before export, and succeeds when a user receives a sourced three-competitor briefing in under ten minutes with at least 80% evidence coverage.

MarketLens AI starts with an apparel sourcing brief: company, target market, product categories, requested comparison fields, and optional known competitors. It creates a structured competitor comparison for details such as MOQ, materials, customization, certifications, production location, lead time, and US delivery. The system keeps unsupported claims as `Unknown`, so it does not turn missing public information into a confident-looking answer.

## User And Surface

The primary user is an Arra Global sourcing, marketing, or business-development team member who needs a quick starting point for competitor analysis. They use the system through a Streamlit web application.

## Workflow And Control Flow

```text
Research brief
  -> Load saved context
  -> Discover competitors
  -> Research each competitor
  -> Index sources for future retrieval
  -> Extract comparable facts
  -> Analyze coverage and compile comparison
  -> Evidence review
  -> Human approval
  -> Save report and retain approved-report context
```

The LangGraph state stores the research brief, competitors, collected sources, extracted facts, comparison rows, evidence coverage, review notes, errors, retrieved history, and integration status. This is an agentic workflow rather than a single model call because it controls multiple steps, tools, evidence checks, recovery behavior, and a human handoff.

## Agents And Tools

| Component | Responsibility | Read / Write |
| --- | --- | --- |
| Context node | Retrieves relevant researcher preferences and prior evidence. | Read |
| Discovery tool | Identifies up to three competitors from the input brief. | Read |
| Research tool | Collects current public web evidence for each competitor. | Read |
| Fact extraction service | Produces field-level facts with evidence or `Unknown`. | Read |
| Analysis node | Builds the comparison table and calculates evidence coverage. | Read |
| Review node | Flags tool failures, fewer than three competitors, and coverage below 80%. | Read |
| Export action | Writes the final report only after the user approves it. | Write, human-approved |

The current code uses LangChain tool wrappers and LangGraph orchestration. Tavily provides live search when configured. Mem0 optionally stores durable user preferences and approved-report summaries. LlamaIndex and Pinecone optionally retain source snippets for retrieval in later research runs. OpenAI structured output is optional and disabled by default during a demo rehearsal.

## Human-in-the-Loop And Safety Limits

All research and analysis steps are autonomous reads. The final report cannot be saved until the user selects the approval checkbox and presses **Save approved report**.

The system should never:

- Present unsupported sourcing facts as verified.
- Send outreach, purchase inventory, or change external supplier data.
- Store a report in Mem0 until the user approves it.
- Treat tool output or retrieved web text as instructions for the workflow.

## Failure Handling

| Failure | Current behavior |
| --- | --- |
| Search provider or competitor page returns no evidence | Records an error and marks affected fields as `Unknown`. |
| Fewer than three competitors are available | Adds an incomplete-report review warning. |
| Evidence coverage is below 80% | Adds a review warning and preserves unsupported fields as follow-up questions. |
| Mem0, Pinecone, or LLM service is unconfigured | Continues in a disabled mode without blocking the research graph. |
| Provider initialization or API call fails | Converts the failure to an `unavailable` integration status and falls back to deterministic extraction where possible. |

## Data And Knowledge Sources

There is no proprietary training dataset. The application works from:

- User-entered apparel sourcing requirements.
- Public web search results returned by Tavily when a Tavily key is supplied.
- Clearly labelled sample evidence when `DEMO_MODE=true` for a repeatable classroom demonstration.
- Optional prior approved-report context in Mem0.
- Optional source snippets indexed in Pinecone through LlamaIndex.

No real competitor claim should be treated as verified until its source URL and quote are visible in the report.

## Implementation Prompt Log

This is a condensed, truthful log of the implementation direction used during vibe coding. It can be included in the required project documentation as evidence of AI coding assistance.

1. "Build the Aro/Arra Global marketing research agent using the code-heavy LangChain and LangGraph approach. Use Mem0, LlamaIndex, Pinecone, and available free credits where appropriate."
2. "Use `/Users/nishokmini/project/market-lens-arra` as the GitHub repository for implementing our project agent."
3. "Create an evidence-first workflow: capture the sourcing brief, discover or accept competitors, collect cited evidence, normalize comparable fields, flag weak evidence, and require approval before saving a report."
4. "Add persistent user memory and a source-knowledge layer without making either service mandatory. Recall relevant sourcing preferences before research, then store an approved report's durable learnings only after human export."
5. "Add optional LLM structured extraction that falls back to deterministic evidence extraction if a model key is absent or a provider call fails."

## Iterations And Learnings

1. The first iteration established a deterministic, demo-safe LangGraph pipeline with sample data so the whole agent could be demonstrated without service credentials.
2. The second iteration separated integrations behind adapters. This made Mem0, Pinecone, and the LLM optional rather than single points of failure.
3. The third iteration added an evidence-coverage score and explicit `Unknown` values. This is more trustworthy than filling a comparison table with model guesses.
4. The final iteration added a structured-output extractor, but keeps it opt-in for demo reliability and cost control.

The main learning is that the hard part is workflow control, not prompt wording: state must track sources, errors, coverage, and approval status. Human approval is most valuable immediately before a write action, while the research steps can remain autonomous.

## Evaluation Criteria

The project will be evaluated with these end-to-end criteria:

- The application returns a comparison of three competitors.
- Every non-`Unknown` field includes a source URL and evidence quote.
- The report reaches at least 80% evidence coverage for the requested fields, or clearly warns the user that it did not.
- The report cannot be saved without human approval.
- A user can complete the briefing flow in under ten minutes.

## Current Build Status

Completed:

- Streamlit intake and report interface.
- LangGraph stateful workflow.
- Discovery, research, fact extraction, analysis, and reviewer stages.
- Approval-gated local report export.
- Optional Mem0 memory adapter.
- Optional LlamaIndex and Pinecone source-retrieval adapter.
- Optional OpenAI structured extraction adapter.
- Automated tests for the demo graph and disabled/failure-safe integration behavior.
- A clearly labelled [three-competitor demo sample output](sample_demo_report.json).

Configuration still needed before a live demo:

- Add `TAVILY_API_KEY` and set `DEMO_MODE=false` for current public web research.
- Add `MEM0_API_KEY` to demonstrate persistent preference recall.
- Add `PINECONE_API_KEY`, `PINECONE_INDEX_NAME`, and `OPENAI_API_KEY` to demonstrate source retrieval.
- Set `USE_LLM_EXTRACTION=true` only if the structured extraction demo is desired.

## Five-Minute Demo Runbook

1. Start the application with `streamlit run app.py` from the repository root.
2. Show the default Arra Global sourcing brief and explain the requested comparison fields.
3. Run the demo-mode workflow and point out the LangGraph stages represented in the output: evidence review, comparison table, cited evidence, and draft recommendation.
4. Explain that unsupported values remain `Unknown` and that low coverage produces a warning.
5. Select the approval checkbox and save the report, explaining that this is the human-in-the-loop boundary.
6. Briefly show `.env.example` and explain how Tavily, Mem0, Pinecone/LlamaIndex, and structured extraction turn on in a live version.

## Submission Checklist

- [ ] Copy this document into a Google Doc and add the final GitHub repository link.
- [ ] Record a live demo of five minutes or less using the runbook above.
- [ ] Commit and push the codebase to GitHub.
- [ ] Attach a sample report JSON or a screenshot of the comparison view.
- [ ] Submit the Google Doc, video link, and GitHub link through the Week 3 form.
