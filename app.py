from __future__ import annotations

import hmac
import json
import os
from secrets import token_urlsafe
from datetime import UTC, datetime

import streamlit as st
from dotenv import load_dotenv

from market_lens.integrations import MemoryService
from market_lens.models import ResearchBrief
from market_lens.workflow import WORKFLOW_STAGE_LABELS, run_research_with_progress


load_dotenv()

SECRET_KEYS = (
    "APP_ACCESS_CODE",
    "OPENAI_API_KEY",
    "OPENAI_MODEL",
    "OPENAI_EMBEDDING_MODEL",
    "OPENAI_BASE_URL",
    "USE_LLM_EXTRACTION",
    "TAVILY_API_KEY",
    "MEM0_API_KEY",
    "MEM0_USER_ID",
    "PINECONE_API_KEY",
    "PINECONE_INDEX_NAME",
    "DEMO_MODE",
    "MAX_RESEARCH_WORKERS",
)


def load_streamlit_secrets() -> None:
    """Expose Streamlit Community Cloud secrets to provider adapters as environment variables."""
    try:
        for key in SECRET_KEYS:
            if key in st.secrets:
                os.environ.setdefault(key, str(st.secrets[key]))
    except FileNotFoundError:
        # Local development can rely on the ignored .env file instead.
        pass


def access_is_allowed() -> bool:
    access_code = os.getenv("APP_ACCESS_CODE")
    if not access_code:
        st.warning("No access code is configured. Do not publish this live-research app publicly with paid API keys.")
        return True
    if st.session_state.get("access_granted"):
        return True
    entered_code = st.sidebar.text_input("Access code", type="password")
    if entered_code and hmac.compare_digest(entered_code, access_code):
        st.session_state.access_granted = True
        st.rerun()
    st.info("Enter the project access code in the sidebar to use live research.")
    return False


def research_user_id() -> str:
    """Avoid allowing one visitor to request another visitor's saved Mem0 context."""
    if configured_id := os.getenv("MEM0_USER_ID"):
        return configured_id
    return st.session_state.setdefault("researcher_id", f"session-{token_urlsafe(16)}")


def create_brief() -> ResearchBrief | None:
    with st.form("research_brief"):
        st.caption("Research memory is isolated to this browser session unless a private MEM0_USER_ID is configured.")
        company_name = st.text_input("Company", value="Arra Global LLC")
        target_market = st.text_input("Target market", value="United States")
        categories = st.text_input(
            "Product categories (comma separated)",
            value="cotton T-shirts, polos, hoodies, uniforms, sportswear",
        )
        requirements = st.text_area(
            "What should be compared?",
            value="MOQ, pricing, materials, customization, certifications, production location, lead time, US delivery",
        )
        competitor_limit = st.number_input(
            "Competitors to research",
            min_value=3,
            max_value=10,
            value=10,
            help="Ten competitors can take several minutes and use more Tavily and model credits.",
        )
        known_competitors = st.text_input("Known competitors (optional, comma separated)")
        submitted = st.form_submit_button("Research competitors", type="primary")

    if not submitted:
        return None

    category_list = [item.strip() for item in categories.split(",") if item.strip()]
    requirement_list = [item.strip() for item in requirements.split(",") if item.strip()]
    competitor_list = [item.strip() for item in known_competitors.split(",") if item.strip()]
    if not company_name or not category_list or not requirement_list:
        st.error("Company, at least one product category, and one comparison field are required.")
        return None

    return ResearchBrief(
        company_name=company_name,
        target_market=target_market,
        product_categories=category_list,
        requested_fields=requirement_list,
        known_competitors=competitor_list,
        competitor_limit=competitor_limit,
        user_id=research_user_id(),
    )


def render_report(state: dict) -> None:
    status = state.get("integration_status", {})
    st.caption(f"Research mode: {state.get('research_mode', 'unknown')}")
    st.caption(
        " | ".join(
            [
                f"Mem0: {status.get('mem0', 'not run')}",
                f"Pinecone retrieval: {status.get('pinecone', 'not run')}",
                f"Pinecone indexing: {status.get('pinecone_indexing', 'not run')}",
                f"LLM extraction: {status.get('llm_extraction', 'not run')}",
            ]
        )
    )
    if state.get("memory_context"):
        with st.expander("Relevant saved preferences"):
            st.json(state["memory_context"])
    if state.get("retrieved_sources"):
        with st.expander("Relevant evidence from earlier runs"):
            st.json(state["retrieved_sources"])

    st.subheader("Evidence review")
    st.metric("Evidence coverage", f"{state['coverage_score']:.0%}")
    if state["review_notes"]:
        for note in state["review_notes"]:
            st.warning(note)

    st.subheader("Competitor comparison")
    st.dataframe(state["comparison_rows"], use_container_width=True, hide_index=True)

    with st.expander("Cited evidence"):
        for fact in state["facts"]:
            st.markdown(f"**{fact['competitor']} - {fact['field']}**: {fact['value']}")
            if fact["evidence"]:
                st.caption(f"{fact['evidence'][0]['quote']}\nSource: {fact['evidence'][0]['url']}")
            else:
                st.caption("No supporting evidence was found.")

    st.subheader("Draft recommendation")
    st.write(state["draft_report"])


def prepare_approved_report(state: dict) -> tuple[str, str, str]:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    filename = f"market-lens-{timestamp}.json"
    report_json = json.dumps(state, indent=2, default=str)
    brief = state["brief"]
    memory_result = MemoryService().remember_approved_report(
        user_id=brief["user_id"],
        brief=brief,
        coverage_score=state["coverage_score"],
    )
    message = "Approved report summary added to Mem0." if memory_result.status == "connected" else memory_result.message
    return filename, report_json, message


def main() -> None:
    st.set_page_config(page_title="MarketLens AI", page_icon="ML", layout="wide")
    load_streamlit_secrets()
    st.title("MarketLens AI")
    st.caption("Arra Global apparel competitor intelligence with evidence-first research and human approval.")

    demo_mode = os.getenv("DEMO_MODE", "true").lower() == "true"
    active_mode = "demo" if demo_mode else "live"
    previous_mode = st.session_state.get("research_mode")
    if previous_mode and previous_mode != active_mode:
        # Prevent a prior demo report from being presented as live research.
        st.session_state.pop("research_state", None)
        st.info("Research mode changed. Submit the brief again to generate a fresh report.")
    st.session_state.research_mode = active_mode
    if demo_mode:
        st.info("Demo mode is active. Results use labelled sample evidence unless you set a Tavily key and turn demo mode off.")
    else:
        st.success("Live research mode is active. MarketLens will use Tavily for competitor discovery and public evidence.")

    if not access_is_allowed():
        return

    brief = create_brief()
    if brief:
        with st.status("Starting MarketLens research...", expanded=True) as status:
            completed_stages: list[str] = []

            def show_stage(stage: str) -> None:
                completed_stages.append(stage)
                status.write(f"{len(completed_stages)}. {WORKFLOW_STAGE_LABELS.get(stage, stage)}")

            try:
                st.session_state.research_state = run_research_with_progress(brief.model_dump(), show_stage)
                st.session_state.research_state["research_mode"] = active_mode
                status.update(label="Research complete", state="complete", expanded=False)
            except Exception as error:
                status.update(label="Research could not complete", state="error", expanded=True)
                st.error(f"MarketLens encountered an unexpected error: {error}")
                return

    state = st.session_state.get("research_state")
    if not state:
        return

    render_report(state)
    approved = st.checkbox("I approve this draft for export", key="approval")
    if st.button("Prepare approved report", disabled=not approved):
        filename, report_json, memory_message = prepare_approved_report(state)
        st.session_state.approved_report = {"filename": filename, "content": report_json, "message": memory_message}
        st.success("Approved report is ready to download.")
    if report := st.session_state.get("approved_report"):
        st.download_button("Download approved report", report["content"], file_name=report["filename"], mime="application/json")
        st.caption(report["message"])


if __name__ == "__main__":
    main()
