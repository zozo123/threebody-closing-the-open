#!/usr/bin/env python3
"""Build or verify the frozen numerical-serialization conformance artifacts."""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
from fractions import Fraction
from pathlib import Path
from typing import Any, Callable

from pydantic import ValidationError

from threebody_atlas.numeric_evidence import (
    SPEC_SHA256,
    BinaryPrecision,
    BinaryValue,
    DecimalPrecision,
    DecimalValue,
    EvidenceDocument,
    EvidenceField,
    ExactPrecision,
    Float64Value,
    HashValue,
    IntegerValue,
    IntervalValue,
    MissingValue,
    NumericEvidenceError,
    Producer,
    RationalValue,
    UnitDescriptor,
    canonical_bytes,
    canonical_sha256,
    decode_value,
    dump_document,
    exact_fraction,
    load_document,
    reference_bytes,
    strict_loads,
)

ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "research/evidence/NUMERIC_SERIALIZATION_SPEC.md"
MATRIX = ROOT / "research/evidence/V1_NUMERIC_ROUNDTRIP_MATRIX_2026-08-16.json"
AUDIT = ROOT / "research/evidence/V1_SERIALIZATION_MUTATION_AUDIT_2026-08-16.json"

DIMENSIONLESS = UnitDescriptor(
    system="dimensionless",
    symbol="1",
    dimension="dimensionless",
)
MASS_RATIO = UnitDescriptor(
    system="atlas_normalized",
    symbol="m2/m3",
    dimension="mass_ratio",
    definition="mass ratio in the frozen m3=1 atlas normalization",
)
TIME = UnitDescriptor(
    system="atlas_normalized",
    symbol="T",
    dimension="time",
    definition="period in the frozen G=1 atlas normalization",
)
PRODUCER = Producer(implementation="threebody_atlas.numeric_evidence", version="v1")


def document(
    value: Any,
    *,
    name: str = "test.value",
    unit: UnitDescriptor = DIMENSIONLESS,
    release_critical: bool = True,
    display: str | None = None,
) -> EvidenceDocument:
    return EvidenceDocument(
        spec_sha256=SPEC_SHA256,
        producer=PRODUCER,
        fields={
            name: EvidenceField(
                value=value,
                unit=unit,
                release_critical=release_critical,
                display=display,
            )
        },
    )


def values() -> list[tuple[str, Any, UnitDescriptor, str | None]]:
    exact = ExactPrecision()
    nearest_binary_200 = BinaryPrecision(bits=200, rounding="nearest_ties_even")
    decimal_60 = DecimalPrecision(digits=60, rounding="nearest_ties_even")
    lower = DecimalValue(
        coefficient="7540187211922449",
        exponent10=-16,
        precision=decimal_60,
    )
    upper = DecimalValue(
        coefficient="7540187227698909",
        exponent10=-16,
        precision=decimal_60,
    )
    return [
        ("float64-one", Float64Value.from_float(1.0), DIMENSIONLESS, "1.000"),
        ("float64-negative-zero", Float64Value.from_float(-0.0), DIMENSIONLESS, "-0"),
        (
            "float64-min-subnormal",
            Float64Value(
                bits="0000000000000001",
                precision=BinaryPrecision(bits=53, rounding="nearest_ties_even"),
            ),
            DIMENSIONLESS,
            "4.9406564584124654e-324",
        ),
        (
            "binary-200-bit",
            BinaryValue(
                significand=str(2**199 + 1),
                exponent2=-199,
                precision=nearest_binary_200,
            ),
            DIMENSIONLESS,
            None,
        ),
        (
            "decimal-60-digit-context",
            DecimalValue(
                coefficient="314159265358979323846264338327950288419716939937510582097494",
                exponent10=-59,
                precision=decimal_60,
            ),
            TIME,
            "3.14159265358979323846264338327950288419716939937510582097494",
        ),
        (
            "large-integer",
            IntegerValue(value=str(2**521 - 1), precision=exact),
            DIMENSIONLESS,
            None,
        ),
        (
            "exact-rational",
            RationalValue(numerator="-355", denominator="113", precision=exact),
            DIMENSIONLESS,
            "-355/113",
        ),
        (
            "closed-decimal-interval",
            IntervalValue(
                lower=lower,
                upper=upper,
                lower_closed=True,
                upper_closed=True,
            ),
            MASS_RATIO,
            "[0.7540187211922449, 0.7540187227698909]",
        ),
        (
            "sha256",
            HashValue(digest=hashlib.sha256(b"threebody-atlas").hexdigest()),
            DIMENSIONLESS,
            None,
        ),
        (
            "unresolved",
            MissingValue(
                reason="precision_insufficient",
                detail="gate cannot be resolved at the declared precision",
            ),
            DIMENSIONLESS,
            None,
        ),
    ]


def _decoded_summary(value: Any) -> dict[str, Any]:
    decoded = decode_value(value)
    if isinstance(decoded, Fraction):
        return {"type": "rational", "numerator": str(decoded.numerator), "denominator": str(decoded.denominator)}
    if isinstance(decoded, float):
        return {
            "type": "float64",
            "bits": Float64Value.from_float(decoded).bits,
            "negative_zero": decoded == 0.0 and math.copysign(1.0, decoded) < 0,
        }
    if isinstance(decoded, tuple):
        lower, upper, lower_closed, upper_closed = decoded
        return {
            "type": "interval",
            "lower": f"{lower.numerator}/{lower.denominator}",
            "upper": f"{upper.numerator}/{upper.denominator}",
            "lower_closed": lower_closed,
            "upper_closed": upper_closed,
        }
    if isinstance(decoded, bytes):
        return {"type": "bytes", "hex": decoded.hex()}
    if decoded is None:
        return {"type": "missing"}
    return {"type": "decimal", "coefficient_exponent": str(decoded)}


def build_matrix() -> dict[str, Any]:
    rows = []
    for name, value, unit, display in values():
        record = document(value, name=f"case.{name}", unit=unit, display=display)
        canonical = dump_document(record)
        reparsed = load_document(canonical)
        reference = reference_bytes(value)
        if canonical_bytes(reparsed) != canonical.encode("utf-8"):
            raise AssertionError(f"Python canonical round-trip changed {name}")
        rows.append(
            {
                "name": name,
                "canonical_json": canonical,
                "canonical_sha256": canonical_sha256(record),
                "compact_reference_base64": base64.b64encode(reference).decode("ascii"),
                "compact_reference_sha256": hashlib.sha256(reference).hexdigest(),
                "decoded": _decoded_summary(value),
                "document": record.model_dump(mode="json", exclude_none=True),
            }
        )
    return {
        "schema": "atlas.numeric-roundtrip-matrix.v1",
        "serialization_schema": "atlas.numeric-evidence.v1",
        "spec_sha256": SPEC_SHA256,
        "generated_on": "2026-08-16",
        "implementations": {
            "python": "threebody_atlas.numeric_evidence",
            "julia": "julia/verify_numeric_evidence.jl",
            "proof_verifier": "not_available; tracked by issue #162",
        },
        "case_count": len(rows),
        "cases": rows,
    }


def _expect_rejection(name: str, action: Callable[[], Any], contains: str) -> dict[str, Any]:
    try:
        action()
    except NumericEvidenceError as exc:
        message = str(exc)
        return {
            "name": name,
            "expected": "rejected",
            "observed": "rejected",
            "passed": contains.lower() in message.lower(),
            "rejection_code": "ATLAS_NUMERIC_PARSER_REJECTED",
            "exception_type": "NumericEvidenceError",
            "diagnostic": message,
        }
    except ValidationError:
        return {
            "name": name,
            "expected": "rejected",
            "observed": "rejected",
            "passed": True,
            "rejection_code": "ATLAS_NUMERIC_SCHEMA_REJECTED",
            "exception_type": "ValidationError",
            "diagnostic": "document rejected by the numeric-evidence schema",
        }
    return {
        "name": name,
        "expected": "rejected",
        "observed": "accepted",
        "passed": False,
        "diagnostic": "mutation was silently accepted",
    }


def _base_decimal_document() -> EvidenceDocument:
    return document(
        DecimalValue(
            coefficient="7540187211922449",
            exponent10=-16,
            precision=DecimalPrecision(digits=60, rounding="nearest_ties_even"),
        ),
        name="event.mass_ratio",
        unit=MASS_RATIO,
    )


def build_audit() -> dict[str, Any]:
    baseline = _base_decimal_document()
    baseline_dict = baseline.model_dump(mode="json", exclude_none=True)
    baseline_json = dump_document(baseline)
    cases: list[dict[str, Any]] = []

    truncated_dict = json.loads(baseline_json)
    truncated_dict["fields"]["event.mass_ratio"]["value"]["coefficient"] = "75401872119224"
    truncated = EvidenceDocument.model_validate(truncated_dict)
    original_value = exact_fraction(baseline.fields["event.mass_ratio"].value)
    truncated_value = exact_fraction(truncated.fields["event.mass_ratio"].value)
    threshold = Fraction(75401872, 100000000)
    cases.append(
        {
            "name": "decimal-truncation",
            "expected": "identity and gate change detected",
            "observed": "identity and gate change detected",
            "passed": canonical_sha256(baseline) != canonical_sha256(truncated)
            and (original_value > threshold) != (truncated_value > threshold),
            "baseline_sha256": canonical_sha256(baseline),
            "mutated_sha256": canonical_sha256(truncated),
        }
    )

    exponent_dict = json.loads(baseline_json)
    exponent_dict["fields"]["event.mass_ratio"]["value"]["exponent10"] = -15
    exponent_mutation = EvidenceDocument.model_validate(exponent_dict)
    cases.append(
        {
            "name": "decimal-exponent-loss",
            "expected": "identity and exact value change detected",
            "observed": "identity and exact value change detected",
            "passed": canonical_sha256(baseline) != canonical_sha256(exponent_mutation)
            and original_value != exact_fraction(exponent_mutation.fields["event.mass_ratio"].value),
        }
    )

    locale_dict = json.loads(baseline_json)
    locale_dict["fields"]["event.mass_ratio"]["value"]["coefficient"] = "754018721,1922449"
    cases.append(
        _expect_rejection(
            "locale-decimal-comma",
            lambda: EvidenceDocument.model_validate(locale_dict),
            "canonical base-10",
        )
    )

    cases.append(
        _expect_rejection(
            "duplicate-object-key",
            lambda: strict_loads('{"kind":"integer","value":"1","value":"2"}'),
            "duplicate",
        )
    )
    cases.append(
        _expect_rejection(
            "raw-json-exponent-parser-difference",
            lambda: strict_loads('{"measurement":1e-300}'),
            "raw JSON floating-point",
        )
    )
    cases.append(
        _expect_rejection(
            "raw-json-nan",
            lambda: strict_loads('{"measurement":NaN}'),
            "non-standard JSON constant",
        )
    )
    cases.append(
        _expect_rejection(
            "raw-json-infinity",
            lambda: strict_loads('{"measurement":Infinity}'),
            "non-standard JSON constant",
        )
    )

    overflow_dict = json.loads(baseline_json)
    overflow_dict["fields"]["event.mass_ratio"]["value"] = {
        "kind": "float64",
        "bits": "7ff0000000000000",
        "precision": {"mode": "binary", "bits": 53, "rounding": "nearest_ties_even"},
    }
    cases.append(
        _expect_rejection(
            "float64-positive-infinity-bits",
            lambda: EvidenceDocument.model_validate(overflow_dict),
            "NaN and infinity",
        )
    )

    nan_dict = json.loads(json.dumps(overflow_dict))
    nan_dict["fields"]["event.mass_ratio"]["value"]["bits"] = "7ff8000000000001"
    cases.append(
        _expect_rejection(
            "float64-nan-payload",
            lambda: EvidenceDocument.model_validate(nan_dict),
            "NaN and infinity",
        )
    )

    positive_zero = document(Float64Value.from_float(0.0))
    negative_zero = document(Float64Value.from_float(-0.0))
    cases.append(
        {
            "name": "signed-zero",
            "expected": "both accepted with distinct identities and signs",
            "observed": "both accepted with distinct identities and signs",
            "passed": canonical_sha256(positive_zero) != canonical_sha256(negative_zero)
            and math.copysign(1.0, decode_value(negative_zero.fields["test.value"].value)) < 0,
            "positive_bits": positive_zero.fields["test.value"].value.bits,
            "negative_bits": negative_zero.fields["test.value"].value.bits,
        }
    )

    subnormal = Float64Value(
        bits="0000000000000001",
        precision=BinaryPrecision(bits=53, rounding="nearest_ties_even"),
    )
    subnormal_doc = document(subnormal)
    cases.append(
        {
            "name": "minimum-subnormal",
            "expected": "accepted and preserved bit-for-bit",
            "observed": "accepted and preserved bit-for-bit",
            "passed": load_document(dump_document(subnormal_doc)).fields["test.value"].value.bits
            == "0000000000000001"
            and exact_fraction(subnormal) > 0,
        }
    )

    reordered = json.dumps(
        baseline_dict,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=False,
    )
    pretty = dump_document(baseline, pretty=True)
    cases.append(
        {
            "name": "pretty-print-and-member-order",
            "expected": "canonical identity unchanged",
            "observed": "canonical identity unchanged",
            "passed": canonical_sha256(load_document(reordered))
            == canonical_sha256(load_document(pretty))
            == canonical_sha256(baseline),
        }
    )

    no_unit = json.loads(baseline_json)
    del no_unit["fields"]["event.mass_ratio"]["unit"]
    cases.append(
        _expect_rejection(
            "critical-field-unit-omission",
            lambda: EvidenceDocument.model_validate(no_unit),
            "unit",
        )
    )

    no_precision = json.loads(baseline_json)
    del no_precision["fields"]["event.mass_ratio"]["value"]["precision"]
    cases.append(
        _expect_rejection(
            "critical-field-precision-omission",
            lambda: EvidenceDocument.model_validate(no_precision),
            "precision",
        )
    )

    wrong_spec = json.loads(baseline_json)
    wrong_spec["spec_sha256"] = "0" * 64
    cases.append(
        _expect_rejection(
            "specification-hash-substitution",
            lambda: EvidenceDocument.model_validate(wrong_spec),
            "frozen serialization spec",
        )
    )

    return {
        "schema": "atlas.serialization-mutation-audit.v1",
        "serialization_schema": "atlas.numeric-evidence.v1",
        "spec_sha256": SPEC_SHA256,
        "generated_on": "2026-08-16",
        "case_count": len(cases),
        "passed": all(case["passed"] for case in cases),
        "cases": cases,
    }


def _render(payload: dict[str, Any]) -> str:
    return json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def _check_or_write(path: Path, text: str, check: bool) -> None:
    if check:
        if not path.exists():
            raise SystemExit(f"missing generated artifact: {path.relative_to(ROOT)}")
        if path.read_text(encoding="utf-8") != text:
            raise SystemExit(f"stale generated artifact: {path.relative_to(ROOT)}")
        return
    path.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    actual_spec_sha = hashlib.sha256(SPEC.read_bytes()).hexdigest()
    if actual_spec_sha != SPEC_SHA256:
        raise SystemExit(
            f"spec digest mismatch: implementation={SPEC_SHA256} actual={actual_spec_sha}"
        )

    matrix = build_matrix()
    audit = build_audit()
    if not audit["passed"]:
        failed = [case["name"] for case in audit["cases"] if not case["passed"]]
        raise SystemExit(f"serialization mutations escaped detection: {', '.join(failed)}")

    _check_or_write(MATRIX, _render(matrix), args.check)
    _check_or_write(AUDIT, _render(audit), args.check)
    verb = "verified" if args.check else "wrote"
    print(f"{verb} {matrix['case_count']} round-trip cases and {audit['case_count']} mutations")


if __name__ == "__main__":
    main()
