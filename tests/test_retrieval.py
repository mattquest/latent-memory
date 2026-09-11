from eval.retrieval import BM25Index


def test_bm25_uses_query_and_corpus_and_limits_source_dominance():
    documents = [
        {"id": "a0", "source_message_id": 1, "text": "Archive amber memory"},
        {"id": "a1", "source_message_id": 1, "text": "Archive amber"},
        {"id": "b", "source_message_id": 2, "text": "Archive violet"},
        {"id": "c", "source_message_id": 3, "text": "Weather forecast rain"},
    ]
    index = BM25Index(documents)
    assert index.search("violet")[0][0]["id"] == "b"
    hits = index.search("archive amber", k=3, max_per_source=1)
    assert {doc["source_message_id"] for doc, _ in hits} == {1, 2, 3}
    assert hits[-1][1] == 0.0  # zero-score tail remains available at larger budgets
    assert index.search("the who should") == [(document, 0.0) for document in documents]
