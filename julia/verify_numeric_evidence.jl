#!/usr/bin/env julia

# Independent Julia conformance verifier for atlas.numeric-evidence.v1.

using JSON3
using SHA

const ROOT = normpath(joinpath(@__DIR__, ".."))
const SPEC = joinpath(ROOT, "research", "evidence", "NUMERIC_SERIALIZATION_SPEC.md")
const MATRIX = joinpath(
    ROOT,
    "research",
    "evidence",
    "V1_NUMERIC_ROUNDTRIP_MATRIX_2026-08-16.json",
)

function check(condition, message)
    condition || error(message)
    return nothing
end

function exact_power_ratio(coefficient::BigInt, radix::BigInt, exponent::Int)
    if exponent >= 0
        return (coefficient * radix^exponent) // BigInt(1)
    end
    return coefficient // radix^(-exponent)
end

function float64_ratio(bits::UInt64)
    sign = iszero(bits >> 63) ? BigInt(1) : BigInt(-1)
    exponent = Int((bits >> 52) & 0x7ff)
    fraction = BigInt(bits & 0x000f_ffff_ffff_ffff)
    check(exponent != 0x7ff, "NaN/infinity escaped the Python validator")
    if exponent == 0
        return exact_power_ratio(sign * fraction, BigInt(2), -1074)
    end
    significand = BigInt(2)^52 + fraction
    return exact_power_ratio(sign * significand, BigInt(2), exponent - 1023 - 52)
end

function exact_ratio(value)
    kind = String(value.kind)
    if kind == "float64"
        bits = parse(UInt64, String(value.bits), base=16)
        return float64_ratio(bits)
    elseif kind == "binary"
        return exact_power_ratio(
            parse(BigInt, String(value.significand)),
            BigInt(2),
            Int(value.exponent2),
        )
    elseif kind == "decimal"
        return exact_power_ratio(
            parse(BigInt, String(value.coefficient)),
            BigInt(10),
            Int(value.exponent10),
        )
    elseif kind == "integer"
        return parse(BigInt, String(value.value)) // BigInt(1)
    elseif kind == "rational"
        return parse(BigInt, String(value.numerator)) // parse(BigInt, String(value.denominator))
    end
    error("$kind is not a numerical endpoint")
end

function verify_value(value)
    kind = String(value.kind)
    if kind == "float64"
        bits = parse(UInt64, String(value.bits), base=16)
        observed = reinterpret(Float64, bits)
        check(isfinite(observed), "float64 payload is not finite")
        check(
            string(reinterpret(UInt64, observed), base=16, pad=16) == String(value.bits),
            "float64 bits did not survive reinterpret",
        )
        check(Int(value.precision.bits) == 53, "float64 precision must be 53 bits")
        exact_ratio(value)
    elseif kind == "binary"
        significand = parse(BigInt, String(value.significand))
        exponent = Int(value.exponent2)
        bits = Int(value.precision.bits)
        expected = exact_ratio(value)
        setprecision(BigFloat, bits) do
            observed = ldexp(BigFloat(significand), exponent)
            check(
                rationalize(BigInt, observed; tol=0) == expected,
                "binary BigFloat reconstruction disagrees with the declared dyadic",
            )
        end
    elseif kind == "decimal"
        coefficient = parse(BigInt, String(value.coefficient))
        exponent = Int(value.exponent10)
        digits = Int(value.precision.digits)
        expected = exact_ratio(value)
        working_bits = ceil(Int, digits * log2(10)) + 16
        setprecision(BigFloat, working_bits) do
            parsed = parse(BigFloat, "$(coefficient)e$(exponent)")
            check(parsed == BigFloat(expected), "decimal reconstruction lost exact value")
        end
    elseif kind == "integer" || kind == "rational"
        exact_ratio(value)
    elseif kind == "interval"
        lower = exact_ratio(value.lower)
        upper = exact_ratio(value.upper)
        check(lower <= upper, "interval lower bound exceeds upper bound")
        if lower == upper
            check(
                Bool(value.lower_closed) && Bool(value.upper_closed),
                "degenerate interval must be closed",
            )
        end
    elseif kind == "hash"
        digest = String(value.digest)
        check(occursin(r"^[0-9a-f]{64}$", digest), "hash digest is not 64 lowercase hex digits")
        check(String(value.algorithm) == "sha256", "hash algorithm must be sha256")
    elseif kind == "missing"
        check(
            String(value.reason) in (
            "not_computed",
            "not_applicable",
            "precision_insufficient",
            "unresolved",
            "invalidated",
            ),
            "missing-value reason is not in the frozen vocabulary",
        )
    else
        error("unknown evidence value kind: $kind")
    end
end

function main()
    matrix = JSON3.read(read(MATRIX, String))
    spec_sha = bytes2hex(sha256(read(SPEC)))
    check(String(matrix.spec_sha256) == spec_sha, "matrix spec digest disagrees with the spec file")
    check(
        String(matrix.serialization_schema) == "atlas.numeric-evidence.v1",
        "matrix schema is not atlas.numeric-evidence.v1",
    )
    check(Int(matrix.case_count) == length(matrix.cases), "matrix case_count disagrees with cases")

    names = String[]
    for row in matrix.cases
        canonical = String(row.canonical_json)
        parsed = JSON3.read(canonical)
        check(JSON3.write(parsed) == canonical, "canonical JSON is not identity-stable")
        check(
            bytes2hex(sha256(canonical)) == String(row.canonical_sha256),
            "canonical JSON digest mismatch",
        )
        check(String(parsed.spec_sha256) == spec_sha, "document spec digest is stale")
        check(
            String(parsed.schema_version) == "atlas.numeric-evidence.v1",
            "document schema is not atlas.numeric-evidence.v1",
        )
        check(length(parsed.fields) == 1, "conformance row must contain exactly one field")
        field = first(values(parsed.fields))
        check(hasproperty(field, :unit), "field is missing unit")
        check(hasproperty(field, :release_critical), "field is missing release_critical")
        verify_value(field.value)
        push!(names, String(row.name))
    end

    expected = Set([
        "float64-one",
        "float64-negative-zero",
        "float64-min-subnormal",
        "binary-200-bit",
        "decimal-60-digit-context",
        "large-integer",
        "exact-rational",
        "closed-decimal-interval",
        "sha256",
        "unresolved",
    ])
    check(Set(names) == expected, "Julia verifier did not see the frozen case names")
    println("verified $(length(names)) canonical numerical evidence cases in Julia")
end

main()
