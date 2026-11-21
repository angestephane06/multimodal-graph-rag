"""Tests that need neither Neo4j nor Ollama."""

from __future__ import annotations

import pytest

from mmgraphrag.agent import UnsafeQuery, clean_cypher, first_number
from mmgraphrag.coco import select_images
from mmgraphrag.evaluate import counting_questions, extraction_scores
from mmgraphrag.extract import normalise_objects

CATEGORIES = ["dog", "bus", "wine glass", "person"]


def test_normalise_maps_plurals_and_deduplicates() -> None:
    raw = ["Dogs", "dog", "buses", "wine glasses", " Person ", "window", ""]
    assert normalise_objects(raw, CATEGORIES) == ["dog", "bus", "wine glass", "person", "window"]
    assert normalise_objects("not a list", CATEGORIES) == []


@pytest.mark.parametrize(
    "text",
    [
        "MATCH (i:Image) RETURN count(i)",
        "```cypher\nMATCH (i:Image)-[:CONTAINS]->(o:Object {name:'dog'}) RETURN count(i);\n```",
    ],
)
def test_clean_cypher_accepts_read_queries(text: str) -> None:
    assert clean_cypher(text).startswith("MATCH")


@pytest.mark.parametrize(
    "text",
    [
        "MATCH (n) DETACH DELETE n",
        "MATCH (i:Image) SET i.caption = 'x' RETURN i",
        "CALL dbms.components()",
        "LOAD CSV FROM 'file:///etc/passwd' AS l RETURN l",
        "Here is the query: MATCH (i) RETURN i",
    ],
)
def test_clean_cypher_refuses_writes_and_prose(text: str) -> None:
    with pytest.raises(UnsafeQuery):
        clean_cypher(text)


def test_image_ids_and_evidence() -> None:
    from mmgraphrag.agent import graph_evidence, image_ids

    rows = [{"count": 3, "example_ids": [10, 11, 10]}, {"i.id": 12, "i.file": "x.jpg"}]
    assert image_ids(rows) == [10, 11, 12]
    text = graph_evidence("MATCH (i:Image) RETURN count(i) AS count", [{"count": 3}])
    assert "count = 3" in text and "MATCH" in text


def test_examples_query_replaces_the_return_clause() -> None:
    from mmgraphrag.agent import examples_query

    q = "MATCH (i:Image)-[:CONTAINS]->(:Object {name: 'dog'})\nRETURN count(DISTINCT i)"
    assert examples_query(q) == (
        "MATCH (i:Image)-[:CONTAINS]->(:Object {name: 'dog'})\nRETURN DISTINCT i.id AS id LIMIT 6"
    )
    assert examples_query("MATCH (o:Object) RETURN count(o)") is None


def test_first_number_skips_booleans() -> None:
    assert first_number([{"ok": True, "n": 7}]) == 7
    assert first_number([{"name": "dog"}]) is None


def test_extraction_scores() -> None:
    truth = {1: {"dog", "person"}, 2: {"bus"}}
    preds = {1: {"dog", "window"}, 2: {"bus", "person"}}
    s = extraction_scores(preds, truth, set(CATEGORIES))
    # "window" is outside the label space and ignored; "person" on image 2 is a false positive.
    assert (s["precision"], s["recall"]) == (round(2 / 3, 4), round(2 / 3, 4))


def test_counting_questions_have_coco_answers() -> None:
    truth = {i: {"dog", "person"} if i % 2 else {"bus", "person"} for i in range(20)}
    qs = counting_questions(truth, n_single=2, n_pairs=2)
    by_q = {q["question"]: q["coco"] for q in qs}
    assert by_q.get("How many images show a person?", 20) == 20
    assert all(q["coco"] >= 3 for q in qs)


def test_selection_is_deterministic() -> None:
    images = {i: {} for i in range(100)}
    assert select_images(images, 10) == select_images(images, 10)
