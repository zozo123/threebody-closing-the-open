# Canonical numerical evidence serialization specification

Status: frozen v1

Schema identifier: `atlas.numeric-evidence.v1`

Normative implementation: `src/threebody_atlas/numeric_evidence.py`

## 1. Scope and invariants

This specification defines the release-facing representation of numerical evidence.
Its purpose is to preserve value, arithmetic semantics, precision, rounding, units,
and unresolved state across Python, Julia, and later proof-verifier boundaries.

The following are normative:

1. A numerical evidence value is never a bare JSON floating-point number.
2. A high-precision value is never converted through Float64 while encoding,
   decoding, comparing, or hashing it.
3. NaN and positive or negative infinity are not evidence values. A failed or
   unresolved computation uses the typed `missing` value and fails release gates
   closed.
4. Every release-critical field has an explicit unit descriptor. Dimensionless is
   written explicitly; it is not represented by an omitted unit.
5. Every inexact numerical value states its radix precision and rounding mode.
   Exact integers and rationals state `{"mode":"exact"}`.
6. A document binds this exact specification by SHA-256 in `spec_sha256`.
7. Duplicate object keys, non-standard JSON constants, and raw JSON decimal or
   exponent tokens are rejected before schema validation.
8. Object member order and insignificant whitespace do not affect identity.

Existing research JSON is not silently reinterpreted as this schema. Producers must
perform an explicit migration and state the original source digest.

## 2. Document envelope

A document is an object with exactly these members:

```json
{
  "fields": {
    "event.bracket": {
      "release_critical": true,
      "unit": {
        "definition": "mass ratio in the frozen m3=1 atlas normalization",
        "dimension": "mass_ratio",
        "symbol": "m2/m3",
        "system": "atlas_normalized"
      },
      "value": {
        "kind": "interval",
        "lower": {"...": "typed endpoint"},
        "lower_closed": true,
        "upper": {"...": "typed endpoint"},
        "upper_closed": true
      }
    }
  },
  "producer": {"implementation": "name", "version": "immutable version"},
  "schema_version": "atlas.numeric-evidence.v1",
  "spec_sha256": "64 lowercase hexadecimal digits"
}
```

Unknown members are errors at every schema level. Field names use the ASCII grammar
`[a-z][a-z0-9_.-]*`. An empty `fields` object is invalid.

`display`, when present on a field, is presentation text only. Consumers must use
`value` for computation. It remains covered by the document hash so that a changed
published display is visible rather than silently detached from its source value.

## 3. Numerical value classes

### 3.1 Float64

```json
{
  "bits": "3ff0000000000000",
  "kind": "float64",
  "precision": {
    "bits": 53,
    "mode": "binary",
    "rounding": "nearest_ties_even"
  }
}
```

`bits` is the exact eight-byte IEEE-754 binary64 payload in big-endian order,
written as 16 lowercase hexadecimal digits. This preserves signed zero and every
finite subnormal. Exponent field `0x7ff` is invalid, regardless of payload.

### 3.2 Arbitrary-precision binary

```json
{
  "exponent2": -199,
  "kind": "binary",
  "precision": {
    "bits": 200,
    "mode": "binary",
    "rounding": "nearest_ties_even"
  },
  "significand": "803469022129495137770981046170581301261101496891396417650689"
}
```

The exact value is `significand * 2^exponent2`. A nonzero significand is odd;
factors of two are absorbed into `exponent2`. Zero is encoded only as significand
`"0"`, exponent zero. `precision.bits` is at least the significand bit length. This
is a lossless MPFR/BigFloat interchange value and does not depend on a decimal
rendering.

### 3.3 Arbitrary-precision decimal

```json
{
  "coefficient": "7540187211922449",
  "exponent10": -16,
  "kind": "decimal",
  "precision": {
    "digits": 60,
    "mode": "decimal",
    "rounding": "nearest_ties_even"
  }
}
```

The exact value is `coefficient * 10^exponent10`. A nonzero coefficient has no
trailing zero; trailing powers of ten are absorbed into the exponent. Zero uses
coefficient `"0"`, exponent zero. Decimal values are exact decimal interchange
values with a declared source arithmetic context. The coefficient cannot contain
more significant digits than `precision.digits`; parsers must not count printed
digits to infer an omitted context.

### 3.4 Exact integer

```json
{"kind":"integer","precision":{"mode":"exact"},"value":"-42"}
```

### 3.5 Exact rational

```json
{
  "denominator": "7",
  "kind": "rational",
  "numerator": "22",
  "precision": {"mode": "exact"}
}
```

The denominator is positive and numerator and denominator are coprime.

### 3.6 Interval

An interval has typed numerical `lower` and `upper` endpoints plus explicit Boolean
closure flags. The lower endpoint must not exceed the upper endpoint under exact
rational comparison. Equal endpoints are valid only when both ends are closed.
NaN, infinity, hashes, and missing values cannot be interval endpoints.

### 3.7 Hash

```json
{"algorithm":"sha256","digest":"...64 lowercase hex...","kind":"hash"}
```

Hashes are typed bytes, not numbers and not path-dependent identifiers.

### 3.8 Missing or unresolved

```json
{
  "detail": "event bracket did not converge at the declared precision",
  "kind": "missing",
  "reason": "precision_insufficient"
}
```

Allowed reasons are `not_computed`, `not_applicable`, `precision_insufficient`,
`unresolved`, and `invalidated`. A release-critical missing field may use only the
last three fail-closed reasons. JSON null is never a numerical evidence value.

## 4. Canonical integer strings

Integer-bearing strings use `-?(0|[1-9][0-9]*)`. A plus sign, leading zero,
whitespace, grouping separator, decimal comma, and `-0` are forbidden. Exponents
are JSON integers because they are bounded schema metadata, not measured values.

## 5. Precision and rounding

Supported rounding identifiers are:

- `nearest_ties_even`
- `nearest_ties_away`
- `toward_zero`
- `toward_positive`
- `toward_negative`
- `away_zero`

Float64 evidence is fixed to 53 significand bits and nearest-ties-even. Arbitrary
binary and decimal values retain the producer's declared context. Changing
precision or rounding changes canonical identity even if the represented exact
value happens to be equal.

## 6. Units

A unit descriptor has `system`, `symbol`, and `dimension`. `system` is one of `si`,
`atlas_normalized`, `dimensionless`, or `other`. Dimensionless must be written as
symbol `1` and dimension `dimensionless`. Atlas-normalized and other units require
a nonempty `definition`, because the symbol alone cannot reproduce the scale.

Units belong to fields, not scalar endpoints, so both endpoints of an interval have
one unambiguous unit. Unit conversion is outside this wire format and must be an
explicit, separately recorded computation.

## 7. Strict JSON parsing

The accepted syntax is RFC 8259 JSON narrowed as follows:

- duplicate object keys are errors;
- `NaN`, `Infinity`, and `-Infinity` are errors;
- raw tokens containing a decimal point or decimal exponent are errors;
- invalid UTF-8 is an error at the byte-decoding boundary;
- unknown schema members and implicit type coercions are errors.

Thus `1.25`, `1e-30`, `"1,25"` in a coefficient, and a repeated `value` key do not
degrade to a nearby value; the entire document is rejected.

## 8. Canonical identity

Canonical bytes are UTF-8 JSON with:

1. object keys sorted lexicographically by their Unicode code points;
2. no insignificant whitespace;
3. non-ASCII characters emitted directly;
4. standard JSON escaping for quotation mark, reverse solidus, and control bytes;
5. no NaN or infinity emission.

The evidence identity is `SHA256(canonical_bytes(document))`. Pretty-printed and
member-reordered copies must parse to the same model and produce the same identity.
Changing a display string, unit, precision, rounding mode, missing reason, or value
changes identity.

The round-trip matrix also records a compact length-prefixed binary reference for
selected values. That reference is a test oracle, not an additional public format.

## 9. Release gate behavior

A release consumer must:

1. strict-parse the bytes;
2. require this schema identifier and the frozen specification digest;
3. validate every field, value, precision descriptor, and unit;
4. compute gates from typed machine values, never from `display`;
5. reject unresolved critical fields;
6. verify artifact identity before using the values.

Reserialization is permitted only if canonical identity and the exact typed values
are unchanged. A gate that changes after an allowed reserialization is a release
blocker.

## 10. Cross-language conformance

`V1_NUMERIC_ROUNDTRIP_MATRIX_2026-08-16.json` is normative conformance data. Python
validates the complete schema, hostile-input audit, compact reference bytes, and
gate invariance. `julia/verify_numeric_evidence.jl` independently checks canonical
JSON parse/write behavior, bit-exact Float64 values, exact integers/rationals,
arbitrary binary values at their declared BigFloat precision, decimal rational
semantics, and interval ordering.

The repository does not yet contain the proof-verifier implementation tracked by
issue `#162`. Until it exists, no artifact may claim proof-verifier round-trip conformance.
The format's integer, rational, binary, and hash components are deliberately chosen
so that a later verifier need not depend on host floating-point parsing.
