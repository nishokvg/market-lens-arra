from market_lens import research_tools


def test_live_discovery_uses_tavily_and_returns_unique_competitors(monkeypatch):
    monkeypatch.setenv("DEMO_MODE", "false")
    monkeypatch.setenv("TAVILY_API_KEY", "test-key")
    captured: dict[str, object] = {}

    def fake_search(query: str, max_results: int):
        captured["query"] = query
        captured["max_results"] = max_results
        return {
            "results": [
                {"title": "Northstar Apparel | Custom Apparel Manufacturer", "url": "https://northstar-apparel.example"},
                {"title": "Summit Uniforms - Corporate Uniform Supplier", "url": "https://summit-uniforms.example"},
                {"title": "Northstar Apparel | About Us", "url": "https://northstar-apparel.example/about"},
                {"title": "Arra Global LLC | Custom Apparel", "url": "https://arra-global.example"},
                {"title": "T-Shirts & Polos Manufacturers in United States | FOURSOURCE", "url": "https://public.foursource.com/manufacturers"},
                {"title": "Custom Sportswear Manufacturer", "url": "https://pine-thread-supply.example"},
            ]
        }

    monkeypatch.setattr(research_tools, "_tavily_search", fake_search)

    competitors = research_tools.discover_competitors.invoke(
        {
            "company_name": "Arra Global LLC",
            "target_market": "United States",
            "product_categories": ["polos", "hoodies"],
        }
    )

    assert competitors == ["Northstar Apparel", "Summit Uniforms", "Pine Thread Supply"]
    assert "United States" in captured["query"]
    assert captured["max_results"] == 20


def test_live_discovery_uses_the_brand_segment_after_a_generic_title(monkeypatch):
    monkeypatch.setenv("DEMO_MODE", "false")
    monkeypatch.setenv("TAVILY_API_KEY", "test-key")

    monkeypatch.setattr(
        research_tools,
        "_tavily_search",
        lambda _query, max_results: {
            "results": [
                {
                    "title": "Custom Company Uniforms | Corporate Apparel | Lands' End Outfitters",
                    "url": "https://business.landsend.com/uniforms",
                },
                {"title": "Zega Apparel: Custom Clothing Manufacturers USA", "url": "https://zegaapparel.com"},
                {"title": "Wooter Apparel | Custom Jerseys", "url": "https://wooterapparel.com"},
            ]
        },
    )

    competitors = research_tools.discover_competitors.invoke(
        {"company_name": "Arra Global LLC", "target_market": "United States", "product_categories": ["uniforms"]}
    )

    assert competitors == ["Lands' End Outfitters", "Zega Apparel", "Wooter Apparel"]


def test_domain_fallback_separates_a_known_industry_suffix():
    assert research_tools._domain_name("https://thygesenapparel.com") == "Thygesen Apparel"


def test_live_discovery_honors_a_ten_competitor_limit(monkeypatch):
    monkeypatch.setenv("DEMO_MODE", "false")
    monkeypatch.setenv("TAVILY_API_KEY", "test-key")
    captured: dict[str, object] = {}

    def fake_search(_query: str, max_results: int):
        captured["max_results"] = max_results
        return {
            "results": [
                {"title": f"Supplier {number} Apparel", "url": f"https://supplier-{number}.example"}
                for number in range(12)
            ]
        }

    monkeypatch.setattr(research_tools, "_tavily_search", fake_search)

    competitors = research_tools.discover_competitors.invoke(
        {
            "company_name": "Arra Global LLC",
            "target_market": "United States",
            "product_categories": ["polos"],
            "competitor_limit": 10,
        }
    )

    assert len(competitors) == 10
    assert captured["max_results"] == 20


def test_live_discovery_requires_a_tavily_key(monkeypatch):
    monkeypatch.setenv("DEMO_MODE", "false")
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)

    competitors = research_tools.discover_competitors.invoke(
        {
            "company_name": "Arra Global LLC",
            "target_market": "United States",
            "product_categories": ["polos"],
        }
    )

    assert competitors == []
