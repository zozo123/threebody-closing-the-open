#!/usr/bin/env julia

# Independently localize published S/U cells with Julia BigFloat + Vern9.
# Python float64 DOP853 cannot push shallow +1 events below ~5e-8; this is the
# Gate B truth path for those cells. Event 2e-8 and closure 1e-7 stay frozen.

include(joinpath(@__DIR__, "verify_critical_points.jl"))

function parse_published_cells(path::String)
    rows = NamedTuple[]
    first = true
    for line in eachline(path)
        s = strip(line)
        isempty(s) && continue
        startswith(s, "#") && continue
        f = split(s, '\t')
        if first
            first = false
            f[1] == "cell_id" || error("expected cell_id header, got $(f[1])")
            continue
        end
        length(f) >= 13 || error("invalid published cell row: $line")
        push!(rows, (
            cell_id=parse(Int, f[1]),
            m1=parse(BigFloat, f[2]),
            m3=parse(BigFloat, f[3]),
            left_m2=parse(BigFloat, f[4]),
            left_x1=parse(BigFloat, f[5]),
            left_v1=parse(BigFloat, f[6]),
            left_v2=parse(BigFloat, f[7]),
            left_period=parse(BigFloat, f[8]),
            right_m2=parse(BigFloat, f[9]),
            right_x1=parse(BigFloat, f[10]),
            right_v1=parse(BigFloat, f[11]),
            right_v2=parse(BigFloat, f[12]),
            right_period=parse(BigFloat, f[13]),
        ))
    end
    isempty(rows) && error("no published cells parsed")
    rows
end

function infer_mode(left, right)
    modes = String[]
    for mode in CRITICAL_EVENT_MODES
        if has_zero_bracket(critical_event(left, mode), critical_event(right, mode))
            push!(modes, mode)
        end
    end
    length(modes) == 1 || error("published cell does not have a unique event: $modes")
    modes[1]
end

function localize_cell(row; tol::BigFloat, target::BigFloat, event_tol::BigFloat, maxiter::Int=24)
    left = corrected_sample(
        row.m1, row.left_m2, row.m3,
        BigFloat[row.left_x1, row.left_v1, row.left_v2, row.left_period];
        tol=tol, target=target,
    )
    right = corrected_sample(
        row.m1, row.right_m2, row.m3,
        BigFloat[row.right_x1, row.right_v1, row.right_v2, row.right_period];
        tol=tol, target=target,
    )
    if left.masses[2] > right.masses[2]
        left, right = right, left
    end
    mode = infer_mode(left, right)
    vlo = critical_event(left, mode)
    vhi = critical_event(right, mode)
    best = abs(vlo) <= abs(vhi) ? left : right
    best_v = critical_event(best, mode)
    iterations = 0
    while abs(best_v) > event_tol && iterations < maxiter
        iterations += 1
        span = right.masses[2] - left.masses[2]
        span > 0 || break
        denom = vhi - vlo
        secant = denom == 0 ? (left.masses[2] + right.masses[2]) / 2 : left.masses[2] - vlo * span / denom
        guard = span / BigFloat(20)
        midm2 = clamp(secant, left.masses[2] + guard, right.masses[2] - guard)
        theta = (midm2 - left.masses[2]) / span
        pguess = (1 - theta) .* left.p .+ theta .* right.p
        mid = corrected_sample(left.masses[1], midm2, left.masses[3], pguess; tol=tol, target=target)
        vmid = critical_event(mid, mode)
        println(
            "cell=", row.cell_id, " iter=", iterations, " mode=", mode,
            " m2=", midm2, " event=", vmid, " closure=", mid.closure, " width=", span,
        )
        if abs(vmid) < abs(best_v)
            best, best_v = mid, vmid
        end
        abs(best_v) <= event_tol && break
        if sign(vmid) == sign(vlo) || vmid == 0
            left, vlo = mid, vmid
        else
            right, vhi = mid, vmid
        end
    end
    passed = abs(best_v) <= event_tol && best.closure <= target
    return (
        cell_id=row.cell_id,
        mode=mode,
        passed=passed,
        event=best_v,
        closure=best.closure,
        iterations=iterations,
        m2=best.masses[2],
        sample=best,
    )
end

function cell_json(result)
    s = result.sample
    "{" *
    "\"cell_id\":$(result.cell_id)," *
    "\"event_mode\":\"$(result.mode)\"," *
    "\"passed\":$(result.passed)," *
    "\"event_value\":\"$(result.event)\"," *
    "\"closure_norm\":\"$(result.closure)\"," *
    "\"refinement_iterations\":$(result.iterations)," *
    "\"m1\":\"$(s.masses[1])\",\"m2\":\"$(result.m2)\",\"m3\":\"$(s.masses[3])\"," *
    "\"x1\":\"$(s.p[1])\",\"v1\":\"$(s.p[2])\",\"v2\":\"$(s.p[3])\",\"period\":\"$(s.p[4])\"," *
    "\"alpha\":\"$(s.alpha)\",\"beta\":\"$(s.beta)\",\"discriminant\":\"$(s.disc)\"" *
    "}"
end

function main_published()
    length(ARGS) >= 2 || error("usage: localize_published_cells.jl SEEDS OUTPUT [DPS] [TOL_EXP] [CLOSURE_EXP] [EVENT_EXP]")
    seed_path, output = ARGS[1], ARGS[2]
    dps = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 40
    tol_exp = length(ARGS) >= 4 ? parse(Int, ARGS[4]) : 22
    closure_exp = length(ARGS) >= 5 ? parse(Int, ARGS[5]) : 12
    event_tol_text = length(ARGS) >= 6 ? String(ARGS[6]) : "2e-8"
    bits = ceil(Int, dps * log2(10)) + 32
    event_tol = parse(BigFloat, event_tol_text)
    event_tol <= parse(BigFloat, "2e-8") || error("refusing to loosen the 2e-8 event gate")

    setprecision(BigFloat, bits) do
        tol = parse(BigFloat, "1e-$(tol_exp)")
        target = parse(BigFloat, "1e-$(closure_exp)")
        rows = parse_published_cells(seed_path)
        results = Any[]
        for row in rows
            println("localizing published cell ", row.cell_id)
            push!(results, localize_cell(row; tol=tol, target=target, event_tol=event_tol))
        end
        failed = [r.cell_id for r in results if !r.passed]
        mkpath(dirname(output))
        open(output, "w") do io
            print(
                io,
                "{\"implementation\":\"independent Julia BigFloat + Vern9 published-cell localizer\",",
                "\"dps\":", dps, ",",
                "\"ode_tolerance\":\"1e-", tol_exp, "\",",
                "\"closure_target\":\"1e-", closure_exp, "\",",
                "\"event_tolerance\":\"", event_tol_text, "\",",
                "\"localized_cells\":", count(r -> r.passed, results), ",",
                "\"missed_cells\":", length(failed), ",",
                "\"failed_cell_ids\":[", join(string.(failed), ","), "],",
                "\"results\":[", join(cell_json.(results), ","), "],",
                "\"claim_status\":\"independent BigFloat localization of published S/U cells; float64 screening remains non-publication\"}\n",
            )
        end
        isempty(failed) || error("BigFloat event/closure gate failed for cells $(failed)")
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main_published()
end
