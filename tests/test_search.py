from types import SimpleNamespace

import app.search.hybrid as hybrid


def test_build_filter_returns_none_without_filters():
    assert hybrid._build_filter() is None


def test_build_filter_includes_requested_fields():
    query_filter = hybrid._build_filter(
        bid_id="Bid1",
        doc_type="addendum",
        addendum_number=2,
    )

    conditions = {condition.key: condition for condition in query_filter.must}

    assert conditions["bid_id"].match.value == "Bid1"
    assert conditions["doc_type"].match.value == "addendum"
    assert conditions["addendum_number"].match.value == 2


def test_search_documents_returns_results_with_citations(monkeypatch):
    payload = {
        "text": "The revised deadline is July 9.",
        "bid_id": "Bid1",
        "file_name": "Addendum 2.pdf",
        "doc_type": "addendum",
        "page_number": 1,
        "addendum_number": 2,
    }
    fake_point = SimpleNamespace(payload=payload, score=0.9)
    fake_client = SimpleNamespace(
        query_points=lambda **kwargs: SimpleNamespace(points=[fake_point])
    )

    monkeypatch.setattr(hybrid, "get_client", lambda: fake_client)
    monkeypatch.setattr(hybrid, "embed_query", lambda query: [0.1, 0.2])
    monkeypatch.setattr(
        hybrid,
        "embed_keyword_query",
        lambda query: hybrid.models.SparseVector(indices=[1], values=[1.0]),
    )
    monkeypatch.setattr(
        hybrid,
        "rerank_documents",
        lambda query, documents, top_k: documents[:top_k],
    )

    results = hybrid.search_documents(
        query="What is the revised deadline?",
        bid_id="Bid1",
        top_k=1,
    )

    assert len(results) == 1
    assert results[0]["text"] == "The revised deadline is July 9."
    assert results[0]["citation"] == {
        "file": "Addendum 2.pdf",
        "page": 1,
    }


def test_search_documents_returns_empty_list_when_no_matches(monkeypatch):
    fake_client = SimpleNamespace(
        query_points=lambda **kwargs: SimpleNamespace(points=[])
    )

    monkeypatch.setattr(hybrid, "get_client", lambda: fake_client)
    monkeypatch.setattr(hybrid, "embed_query", lambda query: [0.1, 0.2])
    monkeypatch.setattr(
        hybrid,
        "embed_keyword_query",
        lambda query: hybrid.models.SparseVector(indices=[1], values=[1.0]),
    )
    monkeypatch.setattr(
        hybrid,
        "rerank_documents",
        lambda query, documents, top_k: documents[:top_k],
    )

    results = hybrid.search_documents(
        query="Question with no matching passages",
        bid_id="Bid1",
        top_k=5,
    )

    assert results == []