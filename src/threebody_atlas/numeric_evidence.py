"""Canonical, lossless numerical evidence records.

The release-facing format deliberately does not use JSON floating-point numbers.
Every numerical value carries an explicit representation and precision contract so
that parsing cannot silently pass through binary64 or infer precision from printed
digits.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import struct
from decimal import Decimal
from fractions import Fraction
from math import gcd
from typing import Annotated, Any, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

SCHEMA_VERSION = "atlas.numeric-evidence.v1"
SPEC_SHA256 = "ed5898d8ae6d006d2a4a16c0d396e91757a832f13218e974deab088b868f3cb3"

_CANONICAL_INTEGER = re.compile(r"-?(?:0|[1-9][0-9]*)\Z")
_FIELD_NAME = re.compile(r"[a-z][a-z0-9_.-]*\Z")
_HEX_64 = re.compile(r"[0-9a-f]{16}\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")

RoundingMode = Literal[
    "nearest_ties_even",
    "nearest_ties_away",
    "toward_zero",
    "toward_positive",
    "toward_negative",
    "away_zero",
]


class NumericEvidenceError(ValueError):
    """Raised when evidence is ambiguous, lossy, or non-canonical."""


class _FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class ExactPrecision(_FrozenModel):
    mode: Literal["exact"] = "exact"


class BinaryPrecision(_FrozenModel):
    mode: Literal["binary"] = "binary"
    bits: int = Field(ge=2, le=1_000_000)
    rounding: RoundingMode


class DecimalPrecision(_FrozenModel):
    mode: Literal["decimal"] = "decimal"
    digits: int = Field(ge=1, le=1_000_000)
    rounding: RoundingMode


class Float64Value(_FrozenModel):
    kind: Literal["float64"] = "float64"
    bits: str
    precision: BinaryPrecision

    @model_validator(mode="after")
    def validate_encoding(self) -> Float64Value:
        if not _HEX_64.fullmatch(self.bits):
            raise ValueError("float64 bits must be exactly 16 lowercase hexadecimal digits")
        if self.precision.bits != 53:
            raise ValueError("float64 precision must declare 53 significand bits")
        if self.precision.rounding != "nearest_ties_even":
            raise ValueError("stored float64 values must declare IEEE nearest-ties-even rounding")
        exponent = (int(self.bits, 16) >> 52) & 0x7FF
        if exponent == 0x7FF:
            raise ValueError("NaN and infinity are not numerical evidence values")
        return self

    @classmethod
    def from_float(cls, value: float) -> Float64Value:
        if not math.isfinite(value):
            raise NumericEvidenceError("NaN and infinity are not numerical evidence values")
        bits = struct.pack(">d", value).hex()
        return cls(
            bits=bits,
            precision=BinaryPrecision(bits=53, rounding="nearest_ties_even"),
        )

    def as_float(self) -> float:
        return struct.unpack(">d", bytes.fromhex(self.bits))[0]


class BinaryValue(_FrozenModel):
    """An exact arbitrary-precision dyadic value: significand * 2**exponent2."""

    kind: Literal["binary"] = "binary"
    significand: str
    exponent2: int = Field(ge=-1_000_000, le=1_000_000)
    precision: BinaryPrecision

    @model_validator(mode="after")
    def validate_encoding(self) -> BinaryValue:
        significand = _parse_canonical_integer(self.significand, "binary significand")
        if significand == 0:
            if self.exponent2 != 0:
                raise ValueError("zero binary values must use exponent2=0")
        elif significand % 2 == 0:
            raise ValueError("binary significand must be odd; absorb factors of two into exponent2")
        if abs(significand).bit_length() > self.precision.bits:
            raise ValueError("binary precision is smaller than the stored significand")
        return self


class DecimalValue(_FrozenModel):
    """An exact decimal value: coefficient * 10**exponent10."""

    kind: Literal["decimal"] = "decimal"
    coefficient: str
    exponent10: int = Field(ge=-1_000_000, le=1_000_000)
    precision: DecimalPrecision

    @model_validator(mode="after")
    def validate_encoding(self) -> DecimalValue:
        coefficient = _parse_canonical_integer(self.coefficient, "decimal coefficient")
        if coefficient == 0:
            if self.exponent10 != 0:
                raise ValueError("zero decimal values must use exponent10=0")
        elif self.coefficient.endswith("0"):
            raise ValueError("decimal coefficient must not end in zero; adjust exponent10")
        if len(self.coefficient.lstrip("-")) > self.precision.digits:
            raise ValueError("decimal precision is smaller than the stored coefficient")
        return self


class IntegerValue(_FrozenModel):
    kind: Literal["integer"] = "integer"
    value: str
    precision: ExactPrecision

    @model_validator(mode="after")
    def validate_encoding(self) -> IntegerValue:
        _parse_canonical_integer(self.value, "integer value")
        return self


class RationalValue(_FrozenModel):
    kind: Literal["rational"] = "rational"
    numerator: str
    denominator: str
    precision: ExactPrecision

    @model_validator(mode="after")
    def validate_encoding(self) -> RationalValue:
        numerator = _parse_canonical_integer(self.numerator, "rational numerator")
        denominator = _parse_canonical_integer(self.denominator, "rational denominator")
        if denominator <= 0:
            raise ValueError("rational denominator must be positive")
        if gcd(abs(numerator), denominator) != 1:
            raise ValueError("rational must be reduced to lowest terms")
        return self


class HashValue(_FrozenModel):
    kind: Literal["hash"] = "hash"
    algorithm: Literal["sha256"] = "sha256"
    digest: str

    @model_validator(mode="after")
    def validate_digest(self) -> HashValue:
        if not _SHA256.fullmatch(self.digest):
            raise ValueError("sha256 digest must be 64 lowercase hexadecimal digits")
        return self


class MissingValue(_FrozenModel):
    kind: Literal["missing"] = "missing"
    reason: Literal[
        "not_computed",
        "not_applicable",
        "precision_insufficient",
        "unresolved",
        "invalidated",
    ]
    detail: Annotated[str, StringConstraints(min_length=1, max_length=500)] | None = None


NumericEndpoint: TypeAlias = Annotated[
    Float64Value | BinaryValue | DecimalValue | IntegerValue | RationalValue,
    Field(discriminator="kind"),
]


class IntervalValue(_FrozenModel):
    kind: Literal["interval"] = "interval"
    lower: NumericEndpoint
    upper: NumericEndpoint
    lower_closed: bool
    upper_closed: bool

    @model_validator(mode="after")
    def validate_bounds(self) -> IntervalValue:
        lower = exact_fraction(self.lower)
        upper = exact_fraction(self.upper)
        if lower > upper:
            raise ValueError("interval lower endpoint exceeds upper endpoint")
        if lower == upper and not (self.lower_closed and self.upper_closed):
            raise ValueError("an interval with equal endpoints must be closed at both ends")
        return self


EvidenceValue: TypeAlias = Annotated[
    Float64Value
    | BinaryValue
    | DecimalValue
    | IntegerValue
    | RationalValue
    | IntervalValue
    | HashValue
    | MissingValue,
    Field(discriminator="kind"),
]


class UnitDescriptor(_FrozenModel):
    system: Literal["si", "atlas_normalized", "dimensionless", "other"]
    symbol: Annotated[str, StringConstraints(min_length=1, max_length=100)]
    dimension: Annotated[str, StringConstraints(min_length=1, max_length=100)]
    definition: Annotated[str, StringConstraints(min_length=1, max_length=500)] | None = None

    @model_validator(mode="after")
    def validate_unit(self) -> UnitDescriptor:
        if self.system == "dimensionless" and (self.symbol != "1" or self.dimension != "dimensionless"):
            raise ValueError("dimensionless units must use symbol='1' and dimension='dimensionless'")
        if self.system in {"atlas_normalized", "other"} and self.definition is None:
            raise ValueError(f"{self.system} units require an explicit definition")
        return self


class EvidenceField(_FrozenModel):
    value: EvidenceValue
    unit: UnitDescriptor
    release_critical: bool
    display: Annotated[str, StringConstraints(min_length=1, max_length=500)] | None = None
    description: Annotated[str, StringConstraints(min_length=1, max_length=1000)] | None = None


class Producer(_FrozenModel):
    implementation: Annotated[str, StringConstraints(min_length=1, max_length=200)]
    version: Annotated[str, StringConstraints(min_length=1, max_length=100)]


class EvidenceDocument(_FrozenModel):
    schema_version: Literal[SCHEMA_VERSION] = SCHEMA_VERSION
    spec_sha256: str
    producer: Producer
    fields: dict[str, EvidenceField]

    @model_validator(mode="after")
    def validate_document(self) -> EvidenceDocument:
        if self.spec_sha256 != SPEC_SHA256:
            raise ValueError(
                f"spec_sha256 must bind the frozen serialization spec ({SPEC_SHA256})"
            )
        if not self.fields:
            raise ValueError("evidence document must contain at least one field")
        for name, field in self.fields.items():
            if not _FIELD_NAME.fullmatch(name):
                raise ValueError(f"field name is not canonical ASCII syntax: {name!r}")
            if field.release_critical and isinstance(field.value, MissingValue):
                if field.value.reason not in {"precision_insufficient", "unresolved", "invalidated"}:
                    raise ValueError(
                        f"release-critical missing field {name!r} must fail closed with an "
                        "unresolved, invalidated, or precision-insufficient reason"
                    )
        return self


def _parse_canonical_integer(value: str, label: str) -> int:
    if not _CANONICAL_INTEGER.fullmatch(value) or value == "-0":
        raise ValueError(f"{label} is not a canonical base-10 integer string")
    return int(value)


def exact_fraction(value: NumericEndpoint) -> Fraction:
    """Decode an endpoint without any binary64 intermediate conversion."""

    if isinstance(value, Float64Value):
        return Fraction.from_float(value.as_float())
    if isinstance(value, BinaryValue):
        significand = int(value.significand)
        if value.exponent2 >= 0:
            return Fraction(significand * (2**value.exponent2), 1)
        return Fraction(significand, 2 ** (-value.exponent2))
    if isinstance(value, DecimalValue):
        coefficient = int(value.coefficient)
        if value.exponent10 >= 0:
            return Fraction(coefficient * (10**value.exponent10), 1)
        return Fraction(coefficient, 10 ** (-value.exponent10))
    if isinstance(value, IntegerValue):
        return Fraction(int(value.value), 1)
    if isinstance(value, RationalValue):
        return Fraction(int(value.numerator), int(value.denominator))
    raise TypeError(f"not a numerical endpoint: {type(value).__name__}")


def decode_value(value: EvidenceValue) -> float | Decimal | Fraction | tuple[Any, ...] | bytes | None:
    """Decode to a lossless standard Python representation where one exists."""

    if isinstance(value, Float64Value):
        return value.as_float()
    if isinstance(value, DecimalValue):
        return Decimal(f"{value.coefficient}e{value.exponent10}")
    if isinstance(value, (BinaryValue, IntegerValue, RationalValue)):
        return exact_fraction(value)
    if isinstance(value, IntervalValue):
        return (
            exact_fraction(value.lower),
            exact_fraction(value.upper),
            value.lower_closed,
            value.upper_closed,
        )
    if isinstance(value, HashValue):
        return bytes.fromhex(value.digest)
    if isinstance(value, MissingValue):
        return None
    raise TypeError(f"unsupported evidence value: {type(value).__name__}")


def _reject_json_float(value: str) -> None:
    raise NumericEvidenceError(
        f"raw JSON floating-point token {value!r} is forbidden; use a typed numerical envelope"
    )


def _reject_json_constant(value: str) -> None:
    raise NumericEvidenceError(f"non-standard JSON constant {value!r} is forbidden")


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise NumericEvidenceError(f"duplicate JSON object key {key!r}")
        result[key] = value
    return result


def strict_loads(text: str) -> Any:
    """Parse strict JSON, rejecting duplicate keys, floats, NaN, and infinity."""

    try:
        return json.loads(
            text,
            object_pairs_hook=_reject_duplicate_keys,
            parse_float=_reject_json_float,
            parse_constant=_reject_json_constant,
        )
    except json.JSONDecodeError as exc:
        raise NumericEvidenceError(f"invalid JSON: {exc.msg} at byte {exc.pos}") from exc


def load_document(text: str) -> EvidenceDocument:
    try:
        return EvidenceDocument.model_validate(strict_loads(text))
    except NumericEvidenceError:
        raise
    except ValueError as exc:
        raise NumericEvidenceError(str(exc)) from exc


def _json_ready(value: BaseModel | dict[str, Any]) -> dict[str, Any]:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json", exclude_none=True)
    return value


def _reject_untyped_floats(value: Any, path: str = "$") -> None:
    if isinstance(value, float):
        raise NumericEvidenceError(
            f"untyped Python float at {path} is forbidden; use a typed numerical envelope"
        )
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise NumericEvidenceError(f"JSON object key at {path} is not a string")
            _reject_untyped_floats(item, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _reject_untyped_floats(item, f"{path}[{index}]")


def canonical_bytes(value: BaseModel | dict[str, Any]) -> bytes:
    """Return the one hashable JSON representation for a validated record."""

    ready = _json_ready(value)
    _reject_untyped_floats(ready)
    return json.dumps(
        ready,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def canonical_sha256(value: BaseModel | dict[str, Any]) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def dump_document(document: EvidenceDocument, *, pretty: bool = False) -> str:
    """Serialize validated evidence; pretty form has identical canonical identity."""

    if not pretty:
        return canonical_bytes(document).decode("utf-8")
    return json.dumps(
        document.model_dump(mode="json", exclude_none=True),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        indent=2,
    ) + "\n"


def _length_prefix(data: bytes) -> bytes:
    return len(data).to_bytes(4, "big") + data


def _integer_reference(value: int) -> bytes:
    sign = b"\x01" if value < 0 else b"\x00"
    magnitude = abs(value).to_bytes(max(1, (abs(value).bit_length() + 7) // 8), "big")
    return sign + _length_prefix(magnitude)


def reference_bytes(value: EvidenceValue) -> bytes:
    """Compact independent reference encoding used by the round-trip matrix.

    This is intentionally small and unambiguous, not a second public wire format.
    It gives tests a byte-level oracle that is independent of JSON spelling.
    """

    if isinstance(value, Float64Value):
        return b"F" + bytes.fromhex(value.bits)
    if isinstance(value, BinaryValue):
        return (
            b"B"
            + _integer_reference(int(value.significand))
            + struct.pack(">qI", value.exponent2, value.precision.bits)
            + _length_prefix(value.precision.rounding.encode("ascii"))
        )
    if isinstance(value, DecimalValue):
        return (
            b"D"
            + _integer_reference(int(value.coefficient))
            + struct.pack(">qI", value.exponent10, value.precision.digits)
            + _length_prefix(value.precision.rounding.encode("ascii"))
        )
    if isinstance(value, IntegerValue):
        return b"I" + _integer_reference(int(value.value))
    if isinstance(value, RationalValue):
        return (
            b"R"
            + _integer_reference(int(value.numerator))
            + _integer_reference(int(value.denominator))
        )
    if isinstance(value, IntervalValue):
        flags = bytes([(int(value.lower_closed) << 1) | int(value.upper_closed)])
        lower = reference_bytes(value.lower)
        upper = reference_bytes(value.upper)
        return b"V" + flags + _length_prefix(lower) + _length_prefix(upper)
    if isinstance(value, HashValue):
        return b"H" + bytes.fromhex(value.digest)
    if isinstance(value, MissingValue):
        detail = (value.detail or "").encode("utf-8")
        return b"M" + _length_prefix(value.reason.encode("ascii")) + _length_prefix(detail)
    raise TypeError(f"unsupported evidence value: {type(value).__name__}")
