from __future__ import annotations

import base64
import hashlib
import json
import math
import struct
import subprocess
import sys
from decimal import Decimal
from fractions import Fraction
from pathlib import Path

import pytest
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

EXACT = ExactPrecision()
BINARY_53 = BinaryPrecision(bits=53, rounding="nearest_ties_even")
DECIMAL_60 = DecimalPrecision(digits=60, rounding="nearest_ties_even")
DIMENSIONLESS = UnitDescriptor(
    system="dimensionless",
    symbol="1",
    dimension="dimensionless",
)


def evidence_document(value: object, *, critical: bool = True) -> EvidenceDocument:
    return EvidenceDocument(
        spec_sha256=SPEC_SHA256,
        producer=Producer(implementation="pytest", version="1"),
        fields={
            "test.value": EvidenceField(
                value=value,
                unit=DIMENSIONLESS,
                release_critical=critical,
            )
        },
    )


def test_frozen_spec_digest_is_exact():
    assert hashlib.sha256(SPEC.read_bytes()).hexdigest() == SPEC_SHA256


def test_generated_artifacts_are_current_and_mutations_pass():
    subprocess.run(
        [sys.executable, "scripts/build_numeric_evidence_artifacts.py", "--check"],
        cwd=ROOT,
        check=True,
    )
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    assert audit["passed"] is True
    assert audit["case_count"] == len(audit["cases"]) == 15
    rejected = [case for case in audit["cases"] if case["expected"] == "rejected"]
    assert all(case["rejection_code"].startswith("ATLAS_NUMERIC_") for case in rejected)
    assert all(case["exception_type"] in {"NumericEvidenceError", "ValidationError"} for case in rejected)
    assert all("pydantic" not in case["diagnostic"].lower() for case in rejected)


def test_roundtrip_matrix_replays_canonical_and_reference_identities():
    matrix = json.loads(MATRIX.read_text(encoding="utf-8"))
    assert matrix["spec_sha256"] == SPEC_SHA256
    assert matrix["case_count"] == len(matrix["cases"]) == 10
    for row in matrix["cases"]:
        document = load_document(row["canonical_json"])
        assert canonical_sha256(document) == row["canonical_sha256"]
        assert document.model_dump(mode="json", exclude_none=True) == row["document"]
        value = next(iter(document.fields.values())).value
        reference = reference_bytes(value)
        assert base64.b64encode(reference).decode("ascii") == row["compact_reference_base64"]
        assert hashlib.sha256(reference).hexdigest() == row["compact_reference_sha256"]


@pytest.mark.parametrize(
    "token",
    ["0.0", "1e-300", "-2E+8", "NaN", "Infinity", "-Infinity"],
)
def test_strict_json_rejects_raw_floating_tokens_and_constants(token: str):
    with pytest.raises(NumericEvidenceError):
        strict_loads(f'{{"measurement":{token}}}')


def test_strict_json_rejects_duplicate_keys_before_schema_validation():
    with pytest.raises(NumericEvidenceError, match="duplicate.*value"):
        strict_loads('{"value":"first","value":"second"}')


def test_canonical_writer_rejects_untyped_python_floats():
    with pytest.raises(NumericEvidenceError, match="untyped Python float"):
        canonical_bytes({"measurement": 0.1})


def test_float64_preserves_signed_zero_and_subnormal_bits():
    positive_zero = Float64Value.from_float(0.0)
    negative_zero = Float64Value.from_float(-0.0)
    subnormal = Float64Value(bits="0000000000000001", precision=BINARY_53)

    assert positive_zero.bits == "0000000000000000"
    assert negative_zero.bits == "8000000000000000"
    assert math.copysign(1.0, negative_zero.as_float()) == -1.0
    assert struct.pack(">d", subnormal.as_float()).hex() == subnormal.bits
    assert exact_fraction(subnormal) == Fraction(1, 2**1074)
    assert canonical_sha256(evidence_document(positive_zero)) != canonical_sha256(
        evidence_document(negative_zero)
    )


@pytest.mark.parametrize("bits", ["7ff0000000000000", "fff0000000000000", "7ff8000000000001"])
def test_float64_rejects_infinity_and_nan_payloads(bits: str):
    with pytest.raises(ValidationError, match="NaN and infinity"):
        Float64Value(bits=bits, precision=BINARY_53)


def test_arbitrary_binary_value_decodes_exactly_without_float64():
    significand = 2**199 + 1
    value = BinaryValue(
        significand=str(significand),
        exponent2=-199,
        precision=BinaryPrecision(bits=200, rounding="nearest_ties_even"),
    )
    assert decode_value(value) == Fraction(significand, 2**199)
    assert reference_bytes(value).startswith(b"B")


@pytest.mark.parametrize(
    ("significand", "exponent", "message"),
    [("4", -2, "must be odd"), ("0", -1, "exponent2=0"), ("01", 0, "canonical")],
)
def test_arbitrary_binary_noncanonical_forms_fail(significand: str, exponent: int, message: str):
    with pytest.raises(ValidationError, match=message):
        BinaryValue(
            significand=significand,
            exponent2=exponent,
            precision=BinaryPrecision(bits=64, rounding="nearest_ties_even"),
        )


def test_decimal_decode_does_not_round_in_the_host_decimal_context():
    coefficient = "314159265358979323846264338327950288419716939937510582097494"
    value = DecimalValue(
        coefficient=coefficient,
        exponent10=-59,
        precision=DECIMAL_60,
    )
    decoded = decode_value(value)
    assert isinstance(decoded, Decimal)
    assert decoded.as_tuple().digits == tuple(int(digit) for digit in coefficient)
    assert decoded.as_tuple().exponent == -59
    assert exact_fraction(value) == Fraction(int(coefficient), 10**59)


@pytest.mark.parametrize("coefficient", ["01", "+1", "1,25", "10", "-0", " 1"])
def test_decimal_noncanonical_coefficients_fail(coefficient: str):
    with pytest.raises(ValidationError):
        DecimalValue(coefficient=coefficient, exponent10=0, precision=DECIMAL_60)


def test_exact_integer_and_reduced_rational_roundtrip():
    integer = IntegerValue(value=str(2**1024 - 1), precision=EXACT)
    rational = RationalValue(numerator="-355", denominator="113", precision=EXACT)
    assert decode_value(integer) == Fraction(2**1024 - 1, 1)
    assert decode_value(rational) == Fraction(-355, 113)
    with pytest.raises(ValidationError, match="lowest terms"):
        RationalValue(numerator="2", denominator="4", precision=EXACT)
    with pytest.raises(ValidationError, match="positive"):
        RationalValue(numerator="1", denominator="-2", precision=EXACT)


def test_interval_ordering_uses_exact_values_across_representations():
    lower = RationalValue(numerator="1", denominator="10", precision=EXACT)
    upper = DecimalValue(coefficient="1", exponent10=-1, precision=DECIMAL_60)
    closed = IntervalValue(
        lower=lower,
        upper=upper,
        lower_closed=True,
        upper_closed=True,
    )
    assert decode_value(closed) == (Fraction(1, 10), Fraction(1, 10), True, True)
    with pytest.raises(ValidationError, match="must be closed"):
        IntervalValue(
            lower=lower,
            upper=upper,
            lower_closed=True,
            upper_closed=False,
        )


def test_units_precision_and_spec_binding_are_mandatory():
    raw = json.loads(dump_document(evidence_document(Float64Value.from_float(1.0))))
    del raw["fields"]["test.value"]["unit"]
    with pytest.raises(ValidationError):
        EvidenceDocument.model_validate(raw)

    raw = json.loads(dump_document(evidence_document(Float64Value.from_float(1.0))))
    del raw["fields"]["test.value"]["value"]["precision"]
    with pytest.raises(ValidationError):
        EvidenceDocument.model_validate(raw)

    raw = json.loads(dump_document(evidence_document(Float64Value.from_float(1.0))))
    raw["spec_sha256"] = "0" * 64
    with pytest.raises(ValidationError, match="frozen serialization spec"):
        EvidenceDocument.model_validate(raw)


def test_normalized_and_other_units_require_definitions():
    with pytest.raises(ValidationError, match="explicit definition"):
        UnitDescriptor(system="atlas_normalized", symbol="T", dimension="time")
    with pytest.raises(ValidationError, match="dimensionless units"):
        UnitDescriptor(system="dimensionless", symbol="rad", dimension="angle")


def test_release_critical_missing_values_fail_closed():
    with pytest.raises(ValidationError, match="must fail closed"):
        evidence_document(MissingValue(reason="not_computed"))
    unresolved = evidence_document(MissingValue(reason="unresolved", detail="root not isolated"))
    assert decode_value(unresolved.fields["test.value"].value) is None


def test_hashes_are_typed_and_lowercase():
    digest = hashlib.sha256(b"artifact").hexdigest()
    value = HashValue(digest=digest)
    assert decode_value(value) == bytes.fromhex(digest)
    with pytest.raises(ValidationError):
        HashValue(digest=digest.upper())


def test_pretty_printing_and_member_order_do_not_change_identity():
    document = evidence_document(
        DecimalValue(coefficient="125", exponent10=-3, precision=DECIMAL_60)
    )
    pretty = dump_document(document, pretty=True)
    compact = dump_document(document)
    reordered = json.dumps(document.model_dump(mode="json", exclude_none=True), sort_keys=False)

    assert pretty != compact
    assert canonical_sha256(load_document(pretty)) == canonical_sha256(load_document(compact))
    assert canonical_sha256(load_document(reordered)) == canonical_sha256(document)


def test_display_text_is_not_used_as_the_machine_value_but_is_tamper_evident():
    value = DecimalValue(coefficient="125", exponent10=-3, precision=DECIMAL_60)
    first = EvidenceDocument(
        spec_sha256=SPEC_SHA256,
        producer=Producer(implementation="pytest", version="1"),
        fields={
            "test.value": EvidenceField(
                value=value,
                unit=DIMENSIONLESS,
                release_critical=True,
                display="0.125",
            )
        },
    )
    second = first.model_copy(
        update={
            "fields": {
                "test.value": first.fields["test.value"].model_copy(update={"display": "0,125"})
            }
        }
    )
    assert decode_value(first.fields["test.value"].value) == decode_value(
        second.fields["test.value"].value
    )
    assert canonical_sha256(first) != canonical_sha256(second)
