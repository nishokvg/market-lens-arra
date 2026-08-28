from market_lens import workflow
from market_lens.workflow import _research_worker_count, build_research_graph, run_research_with_progress


def test_demo_workflow_returns_three_competitors(monkeypatch):
    monkeypatch.setenv("DEMO_MODE", "true")
    state = build_research_graph().invoke(
        {
            "brief": {
                "company_name": "Arra Global LLC",
                "target_market": "United States",
                "product_categories": ["polos"],
                "requested_fields": ["MOQ", "materials", "lead time"],
                "known_competitors": [],
            }
        }
    )

    assert len(state["competitors"]) == 3
    assert len(state["comparison_rows"]) == 3
    assert state["coverage_score"] > 0
    assert state["review_notes"]
    assert state["integration_status"] == {
        "mem0": "disabled",
        "pinecone": "disabled",
        "pinecone_indexing": "disabled",
        "llm_extraction": "disabled",
    }


def test_known_competitor_is_preserved_and_discovery_fills_the_remaining_slots(monkeypatch):
    monkeypatch.setenv("DEMO_MODE", "true")
    state = build_research_graph().invoke(
        {
            "brief": {
                "company_name": "Arra Global LLC",
                "target_market": "United States",
                "product_categories": ["polos"],
                "requested_fields": ["MOQ"],
                "known_competitors": ["Known Apparel Co"],
                "competitor_limit": 3,
            }
        }
    )

    assert state["competitors"] == ["Known Apparel Co", "Demo Apparel Supply", "Demo Uniform Works"]


def test_progress_runner_reports_each_workflow_stage(monkeypatch):
    monkeypatch.setenv("DEMO_MODE", "true")
    stages: list[str] = []

    state = run_research_with_progress(
        {
            "company_name": "Arra Global LLC",
            "target_market": "United States",
            "product_categories": ["polos"],
            "requested_fields": ["MOQ"],
            "known_competitors": [],
        },
        stages.append,
    )

    assert state["competitors"] == ["Demo Apparel Supply", "Demo Uniform Works", "Demo Activewear Co"]
    assert stages == ["load_context", "discover", "research", "index_sources", "extract", "analyze", "review"]


def test_discovery_node_honors_a_ten_competitor_request(monkeypatch):
    class FakeDiscoveryTool:
        def invoke(self, _arguments):
            return [f"Supplier {number}" for number in range(12)]

    monkeypatch.setattr(workflow, "discover_competitors", FakeDiscoveryTool())

    state = workflow.discover_node(
        {
            "brief": {
                "company_name": "Arra Global LLC",
                "target_market": "United States",
                "product_categories": ["polos"],
                "requested_fields": ["MOQ"],
                "competitor_limit": 10,
            }
        }
    )

    assert state["competitors"] == [f"Supplier {number}" for number in range(10)]


def test_research_worker_count_falls_back_for_invalid_configuration(monkeypatch):
    monkeypatch.setenv("MAX_RESEARCH_WORKERS", "not-a-number")

    assert _research_worker_count(10) == 3
