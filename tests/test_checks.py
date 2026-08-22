from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest

from dataforge.checks import (
    Batch,
    CheckerInputError,
    Finding,
    Pair,
    banned_pattern,
    build_batch,
    check_teacher_batch,
    fields_present,
    format_findings,
    hash_pinned,
    max_sentences,
    min_words,
    no_extra_keys,
    opening_ngram_cap,
    preserved_literals,
    row_rule,
    summarize,
    unique_normalized,
    untouched_field,
)
from dataforge.guards import paired_counterfactual_exemption

HASH = "sha256:" + "a" * 64
OTHER_HASH = "sha256:" + "b" * 64


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> Path:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8"
    )
    return path


def _record(record_id: str, **extra: Any) -> dict[str, Any]:
    record: dict[str, Any] = {
        "record_id": record_id,
        "final_response": f"the original final for {record_id}",
        "group_id": record_id,
        "source_split": "train",
    }
    record.update(extra)
    return record


def _request(record_id: str, *, final: str | None = None, hash_value: str = HASH) -> dict[str, Any]:
    return {
        "record_id": record_id,
        "immutable_hash": hash_value,
        "fields": {"final_response": final or f"the original final for {record_id}"},
        "allowed_edits": ["final_response"],
        "instructions": "Rewrite only the listed fields.",
    }


def _response(
    record_id: str, *, final: str, hash_value: str = HASH, **extra: Any
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "record_id": record_id,
        "immutable_hash": hash_value,
        "fields": {"final_response": final},
    }
    row.update(extra)
    return row


def _run(
    tmp_path: Path,
    *,
    requests: list[dict[str, Any]],
    responses: list[dict[str, Any]],
    records: list[dict[str, Any]],
    rules: Any = (),
) -> list[Finding]:
    return check_teacher_batch(
        _write_jsonl(tmp_path / "requests.jsonl", requests),
        _write_jsonl(tmp_path / "responses.jsonl", responses),
        records,
        rules,
    )


# --------------------------------------------------------------------------
# input pseudo-rule
# --------------------------------------------------------------------------


def test_a_response_row_without_a_record_id_is_reported_against_unknown(tmp_path: Path) -> None:
    findings = _run(
        tmp_path,
        requests=[_request("r1")],
        responses=[{"immutable_hash": HASH, "fields": {"final_response": "x"}}],
        records=[_record("r1")],
    )
    assert findings == [Finding("<unknown>", "input", "row has no record_id")]


def test_a_blank_record_id_is_treated_as_missing(tmp_path: Path) -> None:
    findings = _run(
        tmp_path,
        requests=[_request("r1")],
        responses=[_response("", final="x")],
        records=[_record("r1")],
    )
    assert findings == [Finding("<unknown>", "input", "row has no record_id")]


def test_a_duplicate_response_row_is_reported_across_two_response_files(tmp_path: Path) -> None:
    first = _write_jsonl(tmp_path / "resp-a.jsonl", [_response("r1", final="one")])
    second = _write_jsonl(tmp_path / "resp-b.jsonl", [_response("r1", final="two")])
    findings = check_teacher_batch(
        _write_jsonl(tmp_path / "requests.jsonl", [_request("r1")]),
        [first, second],
        [_record("r1")],
        (),
    )
    assert findings == [Finding("r1", "input", "duplicate response row")]


def test_request_files_are_concatenated_before_matching(tmp_path: Path) -> None:
    first = _write_jsonl(tmp_path / "req-a.jsonl", [_request("r1")])
    second = _write_jsonl(tmp_path / "req-b.jsonl", [_request("r2")])
    findings = check_teacher_batch(
        [first, second],
        _write_jsonl(
            tmp_path / "responses.jsonl",
            [_response("r1", final="one"), _response("r2", final="two")],
        ),
        [_record("r1"), _record("r2")],
        (),
    )
    assert findings == []


def test_a_response_without_a_matching_request_row_is_reported(tmp_path: Path) -> None:
    findings = _run(
        tmp_path,
        requests=[_request("r1")],
        responses=[_response("r2", final="two")],
        records=[_record("r1"), _record("r2")],
    )
    assert findings == [Finding("r2", "input", "no matching request row")]


def test_a_response_without_a_matching_record_is_reported(tmp_path: Path) -> None:
    findings = _run(
        tmp_path,
        requests=[_request("r1"), _request("r2")],
        responses=[_response("r2", final="two")],
        records=[_record("r1")],
    )
    assert findings == [Finding("r2", "input", "no matching record")]


def test_an_unmatched_response_yields_no_pair_so_rules_never_see_it(tmp_path: Path) -> None:
    batch, findings = build_batch(
        _write_jsonl(tmp_path / "requests.jsonl", [_request("r1")]),
        _write_jsonl(tmp_path / "responses.jsonl", [_response("r2", final="two")]),
        [_record("r1")],
    )
    assert batch.pairs == ()
    assert {finding.detail for finding in findings} == {
        "no matching request row",
        "no matching record",
    }


def test_untouched_records_are_the_records_with_no_response_row(tmp_path: Path) -> None:
    batch, findings = build_batch(
        _write_jsonl(tmp_path / "requests.jsonl", [_request("r1"), _request("r2")]),
        _write_jsonl(tmp_path / "responses.jsonl", [_response("r1", final="one")]),
        [_record("r1"), _record("r2")],
    )
    assert findings == []
    assert [pair.record_id for pair in batch.pairs] == ["r1"]
    assert [record["record_id"] for record in batch.untouched] == ["r2"]
    assert set(batch.records) == {"r1", "r2"}


def test_records_may_be_any_iterable_and_duplicate_ids_collapse(tmp_path: Path) -> None:
    batch, findings = build_batch(
        _write_jsonl(tmp_path / "requests.jsonl", [_request("r1")]),
        _write_jsonl(tmp_path / "responses.jsonl", []),
        (record for record in [_record("r1"), _record("r1", group_id="later")]),
    )
    assert findings == []
    assert len(batch.untouched) == 1
    assert batch.untouched[0]["group_id"] == "later"


def test_a_missing_jsonl_file_raises_checker_input_error(tmp_path: Path) -> None:
    with pytest.raises(CheckerInputError, match="missing JSONL file"):
        check_teacher_batch(tmp_path / "nope.jsonl", tmp_path / "nope.jsonl", [], ())


def test_a_malformed_json_line_raises_a_json_decode_error(tmp_path: Path) -> None:
    """The decoder's own diagnostics beat anything the checker could rewrap;
    both it and CheckerInputError are ValueErrors, so one except covers both."""
    path = tmp_path / "responses.jsonl"
    path.write_text('{"record_id": "r1"\n', encoding="utf-8")
    with pytest.raises(json.JSONDecodeError) as caught:
        check_teacher_batch(_write_jsonl(tmp_path / "requests.jsonl", []), path, [], ())
    assert isinstance(caught.value, ValueError)


def test_a_non_object_jsonl_row_raises_checker_input_error(tmp_path: Path) -> None:
    path = tmp_path / "responses.jsonl"
    path.write_text("[1, 2]\n", encoding="utf-8")
    with pytest.raises(CheckerInputError, match="must be a JSON object"):
        check_teacher_batch(_write_jsonl(tmp_path / "requests.jsonl", []), path, [], ())


def test_content_defects_accumulate_instead_of_raising(tmp_path: Path) -> None:
    findings = _run(
        tmp_path,
        requests=[_request("r1"), _request("r2")],
        responses=[
            _response("r1", final="short", hash_value=OTHER_HASH),
            _response("r2", final="also short"),
        ],
        records=[_record("r1"), _record("r2")],
        rules=(hash_pinned(), min_words("final_response", 7)),
    )
    assert [(finding.record_id, finding.rule) for finding in findings] == [
        ("r1", "hash_pinned"),
        ("r1", "min_words"),
        ("r2", "min_words"),
    ]


def test_findings_are_sorted_and_deduplicated(tmp_path: Path) -> None:
    def noisy(_batch: Batch) -> list[Finding]:
        return [
            Finding("r2", "b", "second"),
            Finding("r1", "b", "first"),
            Finding("r1", "a", "zzz"),
            Finding("r1", "b", "first"),
        ]

    findings = _run(
        tmp_path,
        requests=[_request("r1")],
        responses=[_response("r1", final="a rewritten final response with enough words")],
        records=[_record("r1")],
        rules=(noisy,),
    )
    assert findings == [
        Finding("r1", "a", "zzz"),
        Finding("r1", "b", "first"),
        Finding("r2", "b", "second"),
    ]


def test_row_rule_adapts_a_per_pair_function(tmp_path: Path) -> None:
    def details(pair: Pair) -> list[str]:
        assert pair.record["record_id"] == pair.record_id
        assert pair.request["record_id"] == pair.record_id
        return [f"saw {pair.response['fields']['final_response']}"]

    findings = _run(
        tmp_path,
        requests=[_request("r1")],
        responses=[_response("r1", final="hello")],
        records=[_record("r1")],
        rules=(row_rule("greeting", details),),
    )
    assert findings == [Finding("r1", "greeting", "saw hello")]


# --------------------------------------------------------------------------
# built-in factories
# --------------------------------------------------------------------------


def test_hash_pinned_passes_a_pinned_response_and_flags_a_moved_hash(tmp_path: Path) -> None:
    assert (
        _run(
            tmp_path,
            requests=[_request("r1")],
            responses=[_response("r1", final="fine")],
            records=[_record("r1")],
            rules=(hash_pinned(),),
        )
        == []
    )
    findings = _run(
        tmp_path,
        requests=[_request("r1")],
        responses=[_response("r1", final="fine", hash_value=OTHER_HASH)],
        records=[_record("r1")],
        rules=(hash_pinned(),),
    )
    assert findings == [Finding("r1", "hash_pinned", "immutable_hash mismatch")]


def test_hash_pinned_flags_a_hash_missing_from_the_response(tmp_path: Path) -> None:
    response = _response("r1", final="fine")
    del response["immutable_hash"]
    findings = _run(
        tmp_path,
        requests=[_request("r1")],
        responses=[response],
        records=[_record("r1")],
        rules=(hash_pinned(),),
    )
    assert findings == [Finding("r1", "hash_pinned", "immutable_hash missing on response")]


def test_hash_pinned_flags_a_hash_missing_from_the_request(tmp_path: Path) -> None:
    """Two absent hashes must not compare equal into a clean row."""
    request = _request("r1")
    del request["immutable_hash"]
    response = _response("r1", final="fine")
    del response["immutable_hash"]
    findings = _run(
        tmp_path,
        requests=[request],
        responses=[response],
        records=[_record("r1")],
        rules=(hash_pinned(),),
    )
    assert findings == [
        Finding("r1", "hash_pinned", "immutable_hash missing on request"),
        Finding("r1", "hash_pinned", "immutable_hash missing on response"),
    ]


def test_hash_pinned_flags_a_hash_that_is_not_sha256_hex(tmp_path: Path) -> None:
    findings = _run(
        tmp_path,
        requests=[_request("r1", hash_value="sha256:abc")],
        responses=[_response("r1", final="fine", hash_value="sha256:abc")],
        records=[_record("r1")],
        rules=(hash_pinned(),),
    )
    assert findings == [Finding("r1", "hash_pinned", "immutable_hash is not sha256:<64 hex>")]


def test_hash_pinned_flags_a_non_string_hash(tmp_path: Path) -> None:
    findings = _run(
        tmp_path,
        requests=[_request("r1")],
        responses=[_response("r1", final="fine", immutable_hash=1234)],
        records=[_record("r1")],
        rules=(hash_pinned(),),
    )
    assert findings == [Finding("r1", "hash_pinned", "immutable_hash missing on response")]


def test_fields_present_flags_a_missing_and_a_blank_field(tmp_path: Path) -> None:
    rules = (fields_present(("final_response", "user_text")),)
    assert (
        _run(
            tmp_path,
            requests=[_request("r1")],
            responses=[
                _response("r1", final="fine", fields={"final_response": "a", "user_text": "b"})
            ],
            records=[_record("r1")],
            rules=rules,
        )
        == []
    )
    findings = _run(
        tmp_path,
        requests=[_request("r1")],
        responses=[_response("r1", final="   ")],
        records=[_record("r1")],
        rules=rules,
    )
    assert findings == [
        Finding("r1", "fields_present", "field 'final_response' must be non-empty text"),
        Finding("r1", "fields_present", "missing field 'user_text'"),
    ]


def test_fields_present_flags_a_response_without_a_fields_mapping(tmp_path: Path) -> None:
    findings = _run(
        tmp_path,
        requests=[_request("r1")],
        responses=[{"record_id": "r1", "immutable_hash": HASH, "fields": "nope"}],
        records=[_record("r1")],
        rules=(fields_present(("final_response",)),),
    )
    assert findings == [Finding("r1", "fields_present", "response has no fields mapping")]


def test_no_extra_keys_flags_a_response_carrying_an_unexpected_key(tmp_path: Path) -> None:
    allowed = ("record_id", "immutable_hash", "fields")
    assert (
        _run(
            tmp_path,
            requests=[_request("r1")],
            responses=[_response("r1", final="fine")],
            records=[_record("r1")],
            rules=(no_extra_keys(allowed),),
        )
        == []
    )
    findings = _run(
        tmp_path,
        requests=[_request("r1")],
        responses=[_response("r1", final="fine", notes="hi", score=1)],
        records=[_record("r1")],
        rules=(no_extra_keys(allowed),),
    )
    assert findings == [
        Finding("r1", "no_extra_keys", "unexpected top-level keys: ['notes', 'score']")
    ]


def test_untouched_field_flags_an_edited_user_turn(tmp_path: Path) -> None:
    request = _request("r1")
    request["fields"]["user_text"] = "freeze my card"
    rules = (untouched_field("user_text"),)
    assert (
        _run(
            tmp_path,
            requests=[request],
            responses=[_response("r1", final="fine", fields={"user_text": "freeze my card"})],
            records=[_record("r1")],
            rules=rules,
        )
        == []
    )
    findings = _run(
        tmp_path,
        requests=[request],
        responses=[_response("r1", final="fine", fields={"user_text": "please freeze my card"})],
        records=[_record("r1")],
        rules=rules,
    )
    assert findings == [Finding("r1", "untouched_field", "field 'user_text' must not be edited")]


def test_untouched_field_ignores_a_field_the_response_never_submitted(tmp_path: Path) -> None:
    request = _request("r1")
    request["fields"]["user_text"] = "freeze my card"
    findings = _run(
        tmp_path,
        requests=[request],
        responses=[_response("r1", final="fine")],
        records=[_record("r1")],
        rules=(untouched_field("user_text"),),
    )
    assert findings == []


def test_min_words_counts_normalized_words(tmp_path: Path) -> None:
    rules = (min_words("final_response", 7),)
    assert (
        _run(
            tmp_path,
            requests=[_request("r1")],
            responses=[_response("r1", final="one two three four five six seven")],
            records=[_record("r1")],
            rules=rules,
        )
        == []
    )
    findings = _run(
        tmp_path,
        requests=[_request("r1")],
        responses=[_response("r1", final="one two three -- four!")],
        records=[_record("r1")],
        rules=rules,
    )
    assert findings == [
        Finding("r1", "min_words", "final_response has 4 normalized words, needs 7")
    ]


def test_max_sentences_counts_terminal_punctuation(tmp_path: Path) -> None:
    rules = (max_sentences("final_response", 2),)
    assert (
        _run(
            tmp_path,
            requests=[_request("r1")],
            responses=[_response("r1", final="One thing. Then another.")],
            records=[_record("r1")],
            rules=rules,
        )
        == []
    )
    findings = _run(
        tmp_path,
        requests=[_request("r1")],
        responses=[_response("r1", final="One. Two. Three! Four?")],
        records=[_record("r1")],
        rules=rules,
    )
    assert findings == [Finding("r1", "max_sentences", "final_response has 4 sentences, at most 2")]


def test_banned_pattern_reports_the_first_matching_term(tmp_path: Path) -> None:
    pattern = re.compile(r"\b(?:demo|sandbox)\b", re.IGNORECASE)
    rules = (banned_pattern("final_response", pattern),)
    assert (
        _run(
            tmp_path,
            requests=[_request("r1")],
            responses=[_response("r1", final="Your card is frozen.")],
            records=[_record("r1")],
            rules=rules,
        )
        == []
    )
    findings = _run(
        tmp_path,
        requests=[_request("r1")],
        responses=[_response("r1", final="This Demo runs in a sandbox.")],
        records=[_record("r1")],
        rules=rules,
    )
    assert findings == [
        Finding("r1", "banned_pattern", "final_response contains banned wording 'Demo'")
    ]


def test_banned_pattern_accepts_a_string_pattern(tmp_path: Path) -> None:
    findings = _run(
        tmp_path,
        requests=[_request("r1")],
        responses=[_response("r1", final="a mock envelope")],
        records=[_record("r1")],
        rules=(banned_pattern("final_response", r"\bmock\b"),),
    )
    assert findings == [
        Finding("r1", "banned_pattern", "final_response contains banned wording 'mock'")
    ]


def test_preserved_literals_flags_a_dropped_literal_and_ignores_case(tmp_path: Path) -> None:
    def extract(pair: Pair) -> list[str]:
        return ["1792", "Frozen", "frozen"]

    rules = (preserved_literals("final_response", extract),)
    assert (
        _run(
            tmp_path,
            requests=[_request("r1")],
            responses=[_response("r1", final="Card 1792 is FROZEN now.")],
            records=[_record("r1")],
            rules=rules,
        )
        == []
    )
    findings = _run(
        tmp_path,
        requests=[_request("r1")],
        responses=[_response("r1", final="Your card is frozen now.")],
        records=[_record("r1")],
        rules=rules,
    )
    assert findings == [
        Finding("r1", "preserved_literals", "final_response dropped literal '1792'")
    ]


# --------------------------------------------------------------------------
# unique_normalized
# --------------------------------------------------------------------------


def _final_of(record: Any) -> Any:
    return record.get("final_response")


def test_unique_normalized_flags_two_rewrites_that_collide(tmp_path: Path) -> None:
    findings = _run(
        tmp_path,
        requests=[_request("r1"), _request("r2")],
        responses=[
            _response("r1", final="Your card is frozen."),
            _response("r2", final="your card is frozen"),
        ],
        records=[_record("r1"), _record("r2")],
        rules=(unique_normalized("final_response", record_value=_final_of),),
    )
    assert findings == [
        Finding("r2", "unique_normalized", "final_response duplicates the rewrite of r1"),
    ]


def test_unique_normalized_flags_a_rewrite_colliding_with_an_untouched_record(
    tmp_path: Path,
) -> None:
    findings = _run(
        tmp_path,
        requests=[_request("r1")],
        responses=[_response("r1", final="The original final for r2!")],
        records=[_record("r1"), _record("r2", final_response="the original final for r2")],
        rules=(unique_normalized("final_response", record_value=_final_of),),
    )
    assert findings == [
        Finding("r1", "unique_normalized", "final_response duplicates untouched record r2"),
    ]


def test_unique_normalized_details_name_the_field_so_two_instances_stay_distinct(
    tmp_path: Path,
) -> None:
    """Without the field prefix the two rules' details would be byte-identical
    and `check_teacher_batch`'s dedup would collapse them into one finding."""
    record_ids = ["r1", "r2", "r3"]
    responses = [
        _response(
            record_id,
            final="unused",
            fields={"final_response": "the same final", "user_text": "the same user turn"},
        )
        for record_id in record_ids
    ]
    findings = _run(
        tmp_path,
        requests=[_request(record_id) for record_id in record_ids],
        responses=responses,
        records=[
            _record(record_id, user_text=f"user turn {record_id}") for record_id in record_ids
        ],
        rules=(
            unique_normalized("final_response", record_value=_final_of),
            unique_normalized("user_text", record_value=lambda record: record.get("user_text")),
        ),
    )
    assert findings == [
        Finding("r2", "unique_normalized", "final_response duplicates the rewrite of r1"),
        Finding("r2", "unique_normalized", "user_text duplicates the rewrite of r1"),
        Finding("r3", "unique_normalized", "final_response duplicates the rewrite of r1"),
        Finding("r3", "unique_normalized", "user_text duplicates the rewrite of r1"),
    ]


def test_unique_normalized_flags_every_rewrite_after_the_lowest_record_id(tmp_path: Path) -> None:
    record_ids = ["r3", "r1", "r2"]
    findings = _run(
        tmp_path,
        requests=[_request(record_id) for record_id in record_ids],
        responses=[_response(record_id, final="the same final") for record_id in record_ids],
        records=[_record(record_id) for record_id in record_ids],
        rules=(unique_normalized("final_response", record_value=_final_of),),
    )
    assert findings == [
        Finding("r2", "unique_normalized", "final_response duplicates the rewrite of r1"),
        Finding("r3", "unique_normalized", "final_response duplicates the rewrite of r1"),
    ]


def test_unique_normalized_does_not_flag_two_untouched_records(tmp_path: Path) -> None:
    findings = _run(
        tmp_path,
        requests=[_request("r1")],
        responses=[_response("r1", final="a distinct rewritten final")],
        records=[
            _record("r1"),
            _record("r2", final_response="shared text"),
            _record("r3", final_response="shared text"),
        ],
        rules=(unique_normalized("final_response", record_value=_final_of),),
    )
    assert findings == []


def test_unique_normalized_exempts_a_governed_counterfactual_pair(tmp_path: Path) -> None:
    records = [
        _record(
            "r1",
            final_response="original one",
            pair_id="p1",
            pair_target="freeze",
            history=["asked about the blue card"],
        ),
        _record(
            "r2",
            final_response="original two",
            pair_id="p1",
            pair_target="replace",
            history=["asked about the red card"],
        ),
    ]
    rules = (
        unique_normalized(
            "final_response",
            record_value=_final_of,
            exempt=paired_counterfactual_exemption(),
        ),
    )
    findings = _run(
        tmp_path,
        requests=[_request("r1"), _request("r2")],
        responses=[
            _response("r1", final="I can help with that card."),
            _response("r2", final="I can help with that card."),
        ],
        records=records,
        rules=rules,
    )
    assert findings == []


def test_unique_normalized_still_flags_a_pair_that_is_not_governed(tmp_path: Path) -> None:
    records = [
        _record("r1", pair_id="p1", pair_target="freeze", history=["a"]),
        _record("r2", pair_id="p2", pair_target="freeze", history=["b"]),
    ]
    findings = _run(
        tmp_path,
        requests=[_request("r1"), _request("r2")],
        responses=[
            _response("r1", final="I can help with that card."),
            _response("r2", final="I can help with that card."),
        ],
        records=records,
        rules=(
            unique_normalized(
                "final_response",
                record_value=_final_of,
                exempt=paired_counterfactual_exemption(),
            ),
        ),
    )
    assert findings == [
        Finding("r2", "unique_normalized", "final_response duplicates the rewrite of r1"),
    ]


def test_unique_normalized_stamps_the_split_the_exemption_reads(tmp_path: Path) -> None:
    seen: list[list[Any]] = []

    def spy(bucket: Any) -> bool:
        seen.append([row.get("_split") for row in bucket])
        return False

    _run(
        tmp_path,
        requests=[_request("r1")],
        responses=[_response("r1", final="the original final for r2")],
        records=[
            _record("r1"),
            _record("r2", final_response="the original final for r2", source_split="test"),
        ],
        rules=(unique_normalized("final_response", record_value=_final_of, exempt=spy),),
    )
    assert seen == [["train", "test"]]


def test_unique_normalized_passes_records_carrying_the_rewritten_value(tmp_path: Path) -> None:
    seen: list[list[Any]] = []

    def spy(bucket: Any) -> bool:
        seen.append([row.get("final_response") for row in bucket])
        return True

    _run(
        tmp_path,
        requests=[_request("r1"), _request("r2")],
        responses=[
            _response("r1", final="a shared rewrite"),
            _response("r2", final="a shared rewrite"),
        ],
        records=[_record("r1"), _record("r2")],
        rules=(unique_normalized("final_response", record_value=_final_of, exempt=spy),),
    )
    assert seen == [["a shared rewrite", "a shared rewrite"]]


def test_unique_normalized_accepts_a_split_of_callable(tmp_path: Path) -> None:
    seen: list[list[Any]] = []

    def spy(bucket: Any) -> bool:
        seen.append([row.get("_split") for row in bucket])
        return True

    _run(
        tmp_path,
        requests=[_request("r1"), _request("r2")],
        responses=[
            _response("r1", final="a shared rewrite"),
            _response("r2", final="a shared rewrite"),
        ],
        records=[_record("r1"), _record("r2")],
        rules=(
            unique_normalized(
                "final_response",
                record_value=_final_of,
                exempt=spy,
                split_of=lambda record: "validation",
            ),
        ),
    )
    assert seen == [["validation", "validation"]]


# --------------------------------------------------------------------------
# opening_ngram_cap
# --------------------------------------------------------------------------


def _family_of(pair: Pair) -> str:
    return str(pair.record.get("scenario_family", ""))


def test_opening_ngram_cap_flags_every_member_of_an_over_cap_family(tmp_path: Path) -> None:
    requests = [_request(f"r{index}") for index in range(1, 7)]
    responses = [
        _response(f"r{index}", final="I have frozen the card for you.") for index in range(1, 6)
    ] + [_response("r6", final="Your replacement card is on the way.")]
    records = [_record(f"r{index}", scenario_family="cards") for index in range(1, 7)]
    findings = _run(
        tmp_path,
        requests=requests,
        responses=responses,
        records=records,
        rules=(opening_ngram_cap("final_response", _family_of),),
    )
    assert [finding.record_id for finding in findings] == ["r1", "r2", "r3", "r4", "r5"]
    assert {finding.detail for finding in findings} == {
        "final_response opening 3-gram 'i have frozen' used 5 times in cards"
    }


def test_opening_ngram_cap_leaves_a_family_at_the_cap_alone(tmp_path: Path) -> None:
    requests = [_request(f"r{index}") for index in range(1, 5)]
    responses = [
        _response(f"r{index}", final="I have frozen the card for you.") for index in range(1, 5)
    ]
    records = [_record(f"r{index}", scenario_family="cards") for index in range(1, 5)]
    findings = _run(
        tmp_path,
        requests=requests,
        responses=responses,
        records=records,
        rules=(opening_ngram_cap("final_response", _family_of),),
    )
    assert findings == []


def test_opening_ngram_cap_keys_on_the_family_so_another_family_is_untouched(
    tmp_path: Path,
) -> None:
    requests = [_request(f"r{index}") for index in range(1, 8)]
    responses = [
        _response(f"r{index}", final="I have frozen the card for you.") for index in range(1, 8)
    ]
    records = [
        _record(f"r{index}", scenario_family="cards" if index <= 5 else "transfers")
        for index in range(1, 8)
    ]
    findings = _run(
        tmp_path,
        requests=requests,
        responses=responses,
        records=records,
        rules=(opening_ngram_cap("final_response", _family_of),),
    )
    assert [finding.record_id for finding in findings] == ["r1", "r2", "r3", "r4", "r5"]


def test_opening_ngram_cap_counts_rewritten_rows_only(tmp_path: Path) -> None:
    """An untouched record sharing the opening never contributes to the count."""
    requests = [_request(f"r{index}") for index in range(1, 8)]
    responses = [
        _response(f"r{index}", final="I have frozen the card for you.") for index in range(1, 5)
    ]
    records = [
        _record(f"r{index}", scenario_family="cards", final_response="I have frozen the card.")
        for index in range(1, 8)
    ]
    findings = _run(
        tmp_path,
        requests=requests,
        responses=responses,
        records=records,
        rules=(opening_ngram_cap("final_response", _family_of),),
    )
    assert findings == []


def test_opening_ngram_cap_honours_n_and_max_uses(tmp_path: Path) -> None:
    requests = [_request(f"r{index}") for index in range(1, 4)]
    responses = [
        _response("r1", final="I have frozen the card."),
        _response("r2", final="I have replaced the card."),
        _response("r3", final="Your card is fine."),
    ]
    records = [_record(f"r{index}", scenario_family="cards") for index in range(1, 4)]
    findings = _run(
        tmp_path,
        requests=requests,
        responses=responses,
        records=records,
        rules=(opening_ngram_cap("final_response", _family_of, n=2, max_uses=1),),
    )
    assert [finding.record_id for finding in findings] == ["r1", "r2"]
    assert findings[0].detail == "final_response opening 2-gram 'i have' used 2 times in cards"


# --------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------


def test_format_findings_renders_one_line_per_finding() -> None:
    findings = [Finding("r1", "input", "no matching record"), Finding("r2", "min_words", "short")]
    assert format_findings(findings) == "r1: input: no matching record\nr2: min_words: short"
    assert format_findings([]) == ""


def test_summarize_reports_the_batch_shape(tmp_path: Path) -> None:
    batch, findings = build_batch(
        _write_jsonl(tmp_path / "requests.jsonl", [_request("r1"), _request("r2")]),
        _write_jsonl(tmp_path / "responses.jsonl", [_response("r1", final="one")]),
        [_record("r1"), _record("r2")],
    )
    findings = [*findings, Finding("r1", "min_words", "short"), Finding("r1", "input", "x")]
    assert summarize(batch, findings) == {
        "checked": 1,
        "untouched": 1,
        "violations": 2,
        "records": 2,
        "rules": ["input", "min_words"],
    }


def test_summarize_is_json_serializable(tmp_path: Path) -> None:
    batch = Batch(pairs=(), records={}, untouched=())
    assert json.loads(json.dumps(summarize(batch, [])))["checked"] == 0
