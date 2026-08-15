#!/usr/bin/env julia

# Independently verify candidate mixed (+1,-1) Floquet organizers with Julia
# BigFloat + Vern9. Python dynamics, Jacobians, and corrected candidate states
# are not imported. A float64 candidate supplies only the initial mass/chart
# seed. At every trial mass pair the periodic orbit is independently corrected
# at fixed masses before both smooth Floquet event equations are evaluated.

include(joinpath(@__DIR__, "verify_critical_points.jl"))

function parse_mixed_vertex_seeds(path::String)
    seeds = NamedTuple[]
    first_data = true
    for line in eachline(path)
        s = strip(line)
        isempty(s) && continue
        startswith(s, "#") && continue
        f = split(s, '\t')
        if first_data && f[1] == "name"
            first_data = false
            continue
        end
        first_data = false
        length(f) == 8 || error("invalid mixed-vertex seed row: $line")
        push!(seeds, (
            name=String(f[1]),
            m1=parse(BigFloat,f[2]), m2=parse(BigFloat,f[3]), m3=parse(BigFloat,f[4]),
            x1=parse(BigFloat,f[5]), v1=parse(BigFloat,f[6]), v2=parse(BigFloat,f[7]),
            period=parse(BigFloat,f[8]),
        ))
    end
    isempty(seeds) && error("no mixed-vertex seeds parsed")
    seeds
end

function mixed_events(s)
    BigFloat[
        critical_event(s,"plus_one"),
        critical_event(s,"minus_one"),
    ]
end

function corrected_mixed_sample(m1,m2,m3,pguess;tol,target)
    corrected_sample(m1,m2,m3,pguess;tol=tol,target=target)
end

function mixed_mass_jacobian(center;tol,target,h)
    m1,m2,m3 = center.masses
    J = zeros(BigFloat,2,2)
    for j in 1:2
        if j == 1
            plus = corrected_mixed_sample(m1+h,m2,m3,center.p;tol=tol,target=target)
            minus = corrected_mixed_sample(m1-h,m2,m3,center.p;tol=tol,target=target)
        else
            plus = corrected_mixed_sample(m1,m2+h,m3,center.p;tol=tol,target=target)
            minus = corrected_mixed_sample(m1,m2-h,m3,center.p;tol=tol,target=target)
        end
        J[:,j] .= (mixed_events(plus)-mixed_events(minus))/(2h)
    end
    J
end

# Julia's LAPACK-backed svdvals has no Matrix{BigFloat} method. For a 2x2
# real matrix, compute sigma_max from the stable closed form and recover
# sigma_min from |det(J)| = sigma_max*sigma_min. The determinant quotient
# avoids cancellation when the mass-event Jacobian is strongly conditioned.
function singular_values_2x2(J::AbstractMatrix{T}) where {T<:Real}
    size(J) == (2,2) || error("singular_values_2x2 requires a 2x2 matrix")
    a,b = J[1,1],J[1,2]
    c,d = J[2,1],J[2,2]
    p = hypot(a+d,c-b)
    q = hypot(a-d,c+b)
    sigma_max = (p+q)/2
    sigma_min = iszero(sigma_max) ? zero(T) : abs(a*d-b*c)/sigma_max
    T[sigma_max,sigma_min]
end

function solve_mixed_mass_root(seed,center;tol,target,event_target,h,max_shift,maxiter=10)
    current = center
    seed_mass = BigFloat[seed.m1,seed.m2]
    for iter in 1:maxiter
        events = mixed_events(current)
        en = norm(events)
        println("mixed iter=",iter," masses=",current.masses[1],",",current.masses[2],
                " events=",events[1],",",events[2]," norm=",en,
                " closure=",current.closure)
        en <= event_target && return current,iter

        J = mixed_mass_jacobian(current;tol=tol,target=target,h=h)
        sv = singular_values_2x2(J)
        sv[end] > parse(BigFloat,"1e-30") || error("mixed event mass Jacobian is singular")
        delta = J \ (-events)

        # Keep each Newton step local. A candidate that requires a large jump is
        # scientifically a different organizer and must not be silently adopted.
        step_norm = norm(delta)
        max_step = max_shift/2
        if step_norm > max_step
            delta .*= max_step/step_norm
        end

        accepted = nothing
        accepted_events = BigFloat(Inf)
        for ls in 0:6
            scale = BigFloat(2)^(-ls)
            trial_mass = BigFloat[current.masses[1],current.masses[2]] + scale*delta
            all((parse(BigFloat,"0.5") .< trial_mass) .& (trial_mass .< parse(BigFloat,"1.5"))) || continue
            shift = norm(trial_mass-seed_mass)
            shift <= max_shift || continue
            try
                trial = corrected_mixed_sample(
                    trial_mass[1],trial_mass[2],seed.m3,current.p;
                    tol=tol,target=target,
                )
                trial_en = norm(mixed_events(trial))
                if trial_en < en
                    accepted = trial
                    accepted_events = trial_en
                    break
                end
            catch err
                println("  line-search correction failed scale=",scale," error=",err)
            end
        end
        accepted === nothing && error(
            "mixed mass Newton failed to reduce event norm from $en; Jacobian singular values=$sv"
        )
        println("  accepted event norm=",accepted_events)
        current = accepted
    end
    error("mixed mass solve did not reach event target")
end

function mixed_sample_json(s)
    plus,minus = mixed_events(s)
    a = s.alpha - 4
    b = s.beta - 4*s.alpha + 10
    "{" *
    "\"m1\":\"$(s.masses[1])\",\"m2\":\"$(s.masses[2])\",\"m3\":\"$(s.masses[3])\"," *
    "\"x1\":\"$(s.p[1])\",\"v1\":\"$(s.p[2])\",\"v2\":\"$(s.p[3])\",\"period\":\"$(s.p[4])\"," *
    "\"closure_norm\":\"$(s.closure)\"," *
    "\"alpha\":\"$(s.alpha)\",\"beta\":\"$(s.beta)\",\"discriminant\":\"$(s.disc)\"," *
    "\"physical_a\":\"$a\",\"physical_b\":\"$b\"," *
    "\"plus_one_event\":\"$plus\",\"minus_one_event\":\"$minus\"," *
    "\"trace_roots\":[$(json_complex(s.t1)),$(json_complex(s.t2))]" *
    "}"
end

function mixed_result_json(seed,raw,center,root,iters)
    shift1 = root.masses[1]-seed.m1
    shift2 = root.masses[2]-seed.m2
    shift = sqrt(shift1^2+shift2^2)
    "{" *
    "\"name\":\"$(seed.name)\"," *
    "\"screening_masses\":[\"$(seed.m1)\",\"$(seed.m2)\",\"$(seed.m3)\"]," *
    "\"raw_bigfloat\":" * mixed_sample_json(raw) * "," *
    "\"corrected_seed\":" * mixed_sample_json(center) * "," *
    "\"root\":" * mixed_sample_json(root) * "," *
    "\"mass_shift\":\"$shift\",\"iterations\":$iters" *
    "}"
end

function main_mixed_vertices()
    length(ARGS) >= 2 || error(
        "usage: verify_mixed_vertices.jl SEEDS OUTPUT [DPS] [TOL_EXP] [CLOSURE_EXP] " *
        "[EVENT_EXP] [FD_EXP] [MAX_SHIFT_EXP]"
    )
    seed_path,output = ARGS[1],ARGS[2]
    dps = length(ARGS)>=3 ? parse(Int,ARGS[3]) : 50
    tol_exp = length(ARGS)>=4 ? parse(Int,ARGS[4]) : 25
    closure_exp = length(ARGS)>=5 ? parse(Int,ARGS[5]) : 18
    event_exp = length(ARGS)>=6 ? parse(Int,ARGS[6]) : 12
    fd_exp = length(ARGS)>=7 ? parse(Int,ARGS[7]) : 6
    max_shift_exp = length(ARGS)>=8 ? parse(Int,ARGS[8]) : 4
    bits = ceil(Int,dps*log2(10))+32

    setprecision(BigFloat,bits) do
        tol = parse(BigFloat,"1e-$(tol_exp)")
        target = parse(BigFloat,"1e-$(closure_exp)")
        event_target = parse(BigFloat,"1e-$(event_exp)")
        h = parse(BigFloat,"1e-$(fd_exp)")
        max_shift = parse(BigFloat,"1e-$(max_shift_exp)")
        seeds = parse_mixed_vertex_seeds(seed_path)
        results = String[]

        for seed in seeds
            println("verifying mixed Floquet organizer ",seed.name)
            p0 = BigFloat[seed.x1,seed.v1,seed.v2,seed.period]
            masses = (seed.m1,seed.m2,seed.m3)
            raw = orbit_sample(masses,p0;tol=tol)
            center = corrected_mixed_sample(
                seed.m1,seed.m2,seed.m3,p0;tol=tol,target=target,
            )
            root,iters = solve_mixed_mass_root(
                seed,center;
                tol=tol,target=target,event_target=event_target,h=h,
                max_shift=max_shift,maxiter=10,
            )
            events = mixed_events(root)
            norm(events) <= event_target || error("mixed event target failed for $(seed.name)")
            root.closure <= target || error("mixed closure target failed for $(seed.name)")
            shift = norm(BigFloat[root.masses[1]-seed.m1,root.masses[2]-seed.m2])
            shift <= max_shift || error("mixed organizer moved too far for $(seed.name): $shift")
            abs(root.alpha-4) <= 2event_target || error("alpha mixed-vertex gate failed")
            abs(root.beta-4) <= 8event_target || error("beta mixed-vertex gate failed")
            push!(results,mixed_result_json(seed,raw,center,root,iters))
        end

        mkpath(dirname(output))
        open(output,"w") do io
            print(io,
                "{\"implementation\":\"independent Julia BigFloat + Vern9 + variational QR shooting + 2D mass Newton\",",
                "\"dps\":",dps,",\"ode_tolerance\":\"1e-",tol_exp,"\",",
                "\"closure_target\":\"1e-",closure_exp,"\",",
                "\"event_norm_target\":\"1e-",event_exp,"\",",
                "\"mass_fd_step\":\"1e-",fd_exp,"\",",
                "\"max_mass_shift\":\"1e-",max_shift_exp,"\",",
                "\"results\":[",join(results,","),"],",
                "\"claim_status\":\"independent BigFloat reproduction of candidate mixed (+1,-1) organizer; canonical physical/Jordan/nondegeneracy classification remains a separate gate\"}\n"
            )
        end
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main_mixed_vertices()
end
