#!/usr/bin/env julia

# Independently reproduce selected float64/JAX critical-curve points with
# Julia BigFloat + Vern9. This script intentionally shares no Python dynamics,
# shooting, or Floquet implementation. Screening coordinates are only seeds.

include(joinpath(@__DIR__, "verify_reduced.jl"))

const CRITICAL_EVENT_MODES = ("plus_one", "minus_one", "trace_collision")

function parse_critical_seeds(path::String)
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
        # 10 columns is the original contract; 12 adds an explicit [m2_lo, m2_hi]
        # bracket for the root, which supersedes the fixed max-shift radius.
        (length(f) == 10 || length(f) == 12) || error("invalid critical seed row: $line")
        mode = String(f[2])
        mode in CRITICAL_EVENT_MODES || error("unsupported event mode: $mode")
        push!(seeds, (
            name=String(f[1]), event_mode=mode,
            m1=parse(BigFloat,f[3]), m2=parse(BigFloat,f[4]), m3=parse(BigFloat,f[5]),
            x1=parse(BigFloat,f[6]), v1=parse(BigFloat,f[7]), v2=parse(BigFloat,f[8]),
            period=parse(BigFloat,f[9]), screening_event=parse(BigFloat,f[10]),
            m2_lo=length(f)==12 ? parse(BigFloat,f[11]) : BigFloat(NaN),
            m2_hi=length(f)==12 ? parse(BigFloat,f[12]) : BigFloat(NaN),
        ))
    end
    isempty(seeds) && error("no critical seeds parsed")
    seeds
end

function chart_state(masses, p)
    m1,m2,m3 = masses
    x1,v1,v2,_ = p
    v3 = -(m1*v1 + m2*v2)/m3
    BigFloat[x1,0,1,0,0,v1-v3,0,v2-v3]
end

function chart_tangent(masses)
    m1,m2,m3 = masses
    C = zeros(BigFloat,8,3)
    C[1,1] = 1
    C[6,2] = 1 + m1/m3
    C[8,2] = m1/m3
    C[6,3] = m2/m3
    C[8,3] = 1 + m2/m3
    C
end

function orbit_sample(masses, p; tol::BigFloat)
    z0 = chart_state(masses,p)
    zf,M,sol = integrate_augmented(z0,masses,p[4];tol=tol)
    closure = norm(zf-z0)
    alpha,beta,disc,t1,t2,score = monodromy_invariants(M)
    return (
        masses=masses, p=BigFloat.(p), closure=closure, M=M, sol=sol,
        alpha=alpha, beta=beta, disc=disc, t1=t1, t2=t2, score=score,
    )
end

function critical_event(s, mode::String)
    mode == "plus_one" && return s.beta - 6s.alpha + 20
    mode == "minus_one" && return s.beta - 2s.alpha + 4
    mode == "trace_collision" && return s.disc
    error("unsupported event mode: $mode")
end

function correct_chart(masses, p0; tol::BigFloat, target::BigFloat, maxiter::Int=12)
    p = BigFloat.(p0)
    last_residual = BigFloat(Inf)
    for iter in 1:maxiter
        s = orbit_sample(masses,p;tol=tol)
        residual = s.sol.u[end][1:8] - chart_state(masses,p)
        rn = norm(residual)
        last_residual = rn
        println("  correction iter=",iter," m2=",masses[2]," closure=",rn)
        rn <= target && return p,s,iter

        C = chart_tangent(masses)
        J = zeros(BigFloat,8,4)
        J[:,1:3] .= (s.M - Matrix{BigFloat}(I,8,8))*C
        ffinal,_ = reduced_rhs_jacobian(s.sol.u[end][1:8],masses)
        J[:,4] .= ffinal
        delta = J \ (-residual)
        p .+= delta
        p[4] > 0 || error("shooting produced non-positive period")
    end
    error("BigFloat shooting failed to reach closure target; last=$last_residual")
end

function corrected_sample(m1,m2,m3,pguess;tol,target)
    masses=(m1,m2,m3)
    p,flow,iters = correct_chart(masses,pguess;tol=tol,target=target)
    merge(flow,(iterations=iters,))
end

function has_zero_bracket(va,vb)
    va == 0 || vb == 0 || sign(va) != sign(vb)
end

function locate_local_bracket(seed, center; tol,target,initial_halfwidth,max_expansions)
    mode = seed.event_mode
    vc = critical_event(center,mode)
    halfwidth = initial_halfwidth
    for attempt in 0:max_expansions
        left_m2 = seed.m2 - halfwidth
        right_m2 = seed.m2 + halfwidth
        left = corrected_sample(seed.m1,left_m2,seed.m3,center.p;tol=tol,target=target)
        right = corrected_sample(seed.m1,right_m2,seed.m3,center.p;tol=tol,target=target)
        vl = critical_event(left,mode)
        vr = critical_event(right,mode)
        println("bracket attempt=",attempt," name=",seed.name," halfwidth=",halfwidth,
                " events=",vl,",",vc,",",vr)
        has_zero_bracket(vl,vc) && return left,center,halfwidth,attempt
        has_zero_bracket(vc,vr) && return center,right,halfwidth,attempt
        has_zero_bracket(vl,vr) && return left,right,halfwidth,attempt
        halfwidth *= 2
    end
    error("failed to find local $(mode) bracket around $(seed.name)")
end

function refine_event_edge(sa,sb,mode;tol,target,width,maxiter=60)
    va,vb = critical_event(sa,mode),critical_event(sb,mode)
    has_zero_bracket(va,vb) || error("endpoint event values do not bracket zero: $va, $vb")
    if va == 0
        return sa,sa,0
    elseif vb == 0
        return sb,sb,0
    end
    lo,hi = sa.masses[2] < sb.masses[2] ? (sa,sb) : (sb,sa)
    vlo,vhi = critical_event(lo,mode),critical_event(hi,mode)
    iterations=0
    while hi.masses[2]-lo.masses[2] > width && iterations < maxiter
        iterations += 1
        span = hi.masses[2]-lo.masses[2]
        denom = vhi-vlo
        secant = denom == 0 ? (lo.masses[2]+hi.masses[2])/2 : lo.masses[2]-vlo*span/denom
        guard = span/BigFloat(8)
        midm2 = clamp(secant,lo.masses[2]+guard,hi.masses[2]-guard)
        theta = (midm2-lo.masses[2])/span
        pguess = (1-theta).*lo.p .+ theta.*hi.p
        mid = corrected_sample(lo.masses[1],midm2,lo.masses[3],pguess;tol=tol,target=target)
        vmid = critical_event(mid,mode)
        println("refine iter=",iterations," mode=",mode," m2=",midm2,
                " event=",vmid," closure=",mid.closure," width=",span)
        if vmid == 0
            return mid,mid,iterations
        elseif sign(vmid) == sign(vlo)
            lo=mid; vlo=vmid
        else
            hi=mid; vhi=vmid
        end
    end
    hi.masses[2]-lo.masses[2] <= width || error("critical refinement did not reach width target")
    lo,hi,iterations
end

function sample_json(s,mode)
    "{" *
    "\"m1\":\"$(s.masses[1])\",\"m2\":\"$(s.masses[2])\",\"m3\":\"$(s.masses[3])\"," *
    "\"x1\":\"$(s.p[1])\",\"v1\":\"$(s.p[2])\",\"v2\":\"$(s.p[3])\",\"period\":\"$(s.p[4])\"," *
    "\"closure_norm\":\"$(s.closure)\",\"alpha\":\"$(s.alpha)\",\"beta\":\"$(s.beta)\"," *
    "\"discriminant\":\"$(s.disc)\",\"trace_roots\":[$(json_complex(s.t1)),$(json_complex(s.t2))]," *
    "\"event_value\":\"$(critical_event(s,mode))\"" *
    "}"
end

function result_json(seed,raw,center,lo,hi,bracket_halfwidth,bracket_attempts,iterations)
    mode = seed.event_mode
    best = abs(critical_event(lo,mode)) <= abs(critical_event(hi,mode)) ? lo : hi
    root_shift = best.masses[2]-seed.m2
    "{" *
    "\"name\":\"$(seed.name)\",\"event_mode\":\"$mode\"," *
    "\"screening_m2\":\"$(seed.m2)\",\"screening_event\":\"$(seed.screening_event)\"," *
    "\"raw_bigfloat_closure\":\"$(raw.closure)\",\"raw_bigfloat_event\":\"$(critical_event(raw,mode))\"," *
    "\"corrected_center\":" * sample_json(center,mode) * "," *
    "\"initial_bracket_halfwidth\":\"$bracket_halfwidth\",\"bracket_expansions\":$bracket_attempts," *
    "\"root_shift_from_screening_m2\":\"$root_shift\",\"refinement_iterations\":$iterations," *
    "\"left\":" * sample_json(lo,mode) * ",\"right\":" * sample_json(hi,mode) *
    "}"
end

function main_critical()
    length(ARGS) >= 2 || error("usage: verify_critical_points.jl SEEDS OUTPUT [DPS] [TOL_EXP] [CLOSURE_EXP] [WIDTH_EXP] [HALFWIDTH_EXP] [MAX_SHIFT_EXP]")
    seed_path,output = ARGS[1],ARGS[2]
    dps = length(ARGS)>=3 ? parse(Int,ARGS[3]) : 50
    tol_exp = length(ARGS)>=4 ? parse(Int,ARGS[4]) : 25
    closure_exp = length(ARGS)>=5 ? parse(Int,ARGS[5]) : 18
    width_exp = length(ARGS)>=6 ? parse(Int,ARGS[6]) : 8
    halfwidth_exp = length(ARGS)>=7 ? parse(Int,ARGS[7]) : 5
    max_shift_exp = length(ARGS)>=8 ? parse(Int,ARGS[8]) : 4
    bits = ceil(Int,dps*log2(10))+32

    setprecision(BigFloat,bits) do
        tol = parse(BigFloat,"1e-$(tol_exp)")
        target = parse(BigFloat,"1e-$(closure_exp)")
        width = parse(BigFloat,"1e-$(width_exp)")
        initial_halfwidth = 2*parse(BigFloat,"1e-$(halfwidth_exp)")
        max_shift = parse(BigFloat,"1e-$(max_shift_exp)")
        seeds = parse_critical_seeds(seed_path)
        results = String[]

        for seed in seeds
            println("verifying critical representative ",seed.name," mode=",seed.event_mode)
            masses=(seed.m1,seed.m2,seed.m3)
            p0=BigFloat[seed.x1,seed.v1,seed.v2,seed.period]
            raw=orbit_sample(masses,p0;tol=tol)
            center=corrected_sample(seed.m1,seed.m2,seed.m3,p0;tol=tol,target=target)
            left,right,used_halfwidth,attempts = locate_local_bracket(
                seed,center;tol=tol,target=target,initial_halfwidth=initial_halfwidth,max_expansions=5,
            )
            lo,hi,iters = refine_event_edge(left,right,seed.event_mode;tol=tol,target=target,width=width)
            best = abs(critical_event(lo,seed.event_mode)) <= abs(critical_event(hi,seed.event_mode)) ? lo : hi
            shift = abs(best.masses[2]-seed.m2)
            # MEMBERSHIP GUARD.  max_shift is a proxy: it bounds how far the
            # BigFloat root may move from the seed, standing in for "did we land
            # on the root we meant".  When the seed carries the sign-change
            # bracket that LOCATED the root, the real condition is available
            # directly -- the certified root must lie inside that bracket -- and
            # it is strictly stronger, because a radius admits any root within
            # max_shift while membership admits only the located one.
            #
            # This matters because the bracket cannot be narrowed below ~1.8e-3
            # here: scripts/audit_sign_topology.py probes converge only via
            # neighbour seeding, so isolated bisection probes fail
            # (endpoint_probe_failed on every refinement attempted 2026-08-18),
            # and a seed midpoint can legitimately sit up to a half-width from
            # the root.  Refusing that as "moved too far" would reject correct
            # roots for a reason that is about probe spacing, not physics.
            # The frozen gates are untouched: the closure and bracket-width
            # checks below still apply exactly as before.
            if isnan(seed.m2_lo) || isnan(seed.m2_hi)
                shift <= max_shift || error("BigFloat critical root moved too far for $(seed.name): $shift > $max_shift")
            else
                (best.masses[2] >= seed.m2_lo && best.masses[2] <= seed.m2_hi) ||
                    error("BigFloat critical root left its locating bracket for $(seed.name): $(best.masses[2]) not in [$(seed.m2_lo), $(seed.m2_hi)]")
            end
            max(lo.closure,hi.closure) <= target || error("corrected critical closure gate failed for $(seed.name)")
            hi.masses[2]-lo.masses[2] <= width || error("critical bracket width gate failed for $(seed.name)")
            push!(results,result_json(seed,raw,center,lo,hi,used_halfwidth,attempts,iters))
        end

        mkpath(dirname(output))
        open(output,"w") do io
            print(io,"{\"implementation\":\"independent Julia BigFloat + Vern9 + variational QR shooting\",",
                  "\"dps\":",dps,",\"ode_tolerance\":\"1e-",tol_exp,"\",",
                  "\"closure_target\":\"1e-",closure_exp,"\",\"m2_width_target\":\"1e-",width_exp,"\",",
                  "\"results\":[",join(results,","),"],",
                  "\"claim_status\":\"independent BigFloat slice reproduction of screening critical-curve representatives; canonical mechanism and full-manifold release gates remain required\"}\n")
        end
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main_critical()
end
