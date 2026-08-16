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
    @assert exponent != 0x7ff "NaN/infinity escaped the Python validator"
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
        @assert isfinite(observed)
        @assert string(reinterpret(UInt64, observed), base=16, pad=16) == String(value.bits)
        @assert Int(value.precision.bits) == 53
        exact_ratio(value)
    elseif kind == "binary"
        significand = parse(BigInt, String(value.significand))
        exponent = Int(value.exponent2)
        bits = Int(value.precision.bits)
        expected = exact_ratio(value)
        setprecision(BigFloat, bits) do
            observed = ldexp(BigFloat(significand), exponent)
            @assert rationalize(BigInt, observed; tol=0) == expected
        end
    elseif kind == "decimal"
        coefficient = parse(BigInt, String(value.coefficient))
        exponent = Int(value.exponent10)
        digits = Int(value.precision.digits)
        expected = exact_ratio(value)
        working_bits = ceil(Int, digits * log2(10)) + 16
        setprecision(BigFloat, working_bits) do
            parsed = parse(BigFloat, "$(coefficient)e$(exponent)")
            @assert parsed == BigFloat(expected)
        end
    elseif kind == "integer" || kind == "rational"
        exact_ratio(value)
    elseif kind == "interval"
        lower = exact_ratio(value.lower)
        upper = exact_ratio(value.upper)
        @assert lower <= upper
        if lower == upper
            @assert Bool(value.lower_closed) && Bool(value.upper_closed)
        end
    elseif kind == "hash"
        digest = String(value.digest)
        @assert occursin(r"^[0-9a-f]{64}$", digest)
        @assert String(value.algorithm) == "sha256"
    elseif kind == "missing"
        @assert String(value.reason) in (
            "not_computed",
            "not_applicable",
            "precision_insufficient",
            "unresolved",
            "invalidated",
        )
    else
        error("unknown evidence value kind: $kind")
    end
end

function main()
    matrix = JSON3.read(read(MATRIX, String))
    spec_sha = bytes2hex(sha256(read(SPEC)))
    @assert String(matrix.spec_sha256) == spec_sha
    @assert String(matrix.serialization_schema) == "atlas.numeric-evidence.v1"
    @assert Int(matrix.case_count) == length(matrix.cases)

    names = String[]
    for row in matrix.cases
        canonical = String(row.canonical_json)
        parsed = JSON3.read(canonical)
        @assert JSON3.write(parsed) == canonical
        @assert bytes2hex(sha256(canonical)) == String(row.canonical_sha256)
        @assert String(parsed.spec_sha256) == spec_sha
        @assert String(parsed.schema_version) == "atlas.numeric-evidence.v1"
        @assert length(parsed.fields) == 1
        field = first(values(parsed.fields))
        @assert hasproperty(field, :unit)
        @assert hasproperty(field, :release_critical)
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
    @assert Set(names) == expected
    println("verified $(length(names)) canonical numerical evidence cases in Julia")
end

main()
