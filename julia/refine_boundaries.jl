#!/usr/bin/env julia

include(joinpath(@__DIR__, "verify_reduced.jl"))

const EVENT_MODES = ("plus_one", "minus_one", "trace_collision")

"""Return the four published-grid rows that bracket the first stable island at m1=0.8."""
function published_brackets(path::String)
    wanted = Set([6, 7, 11, 12])
    rows = parse_baseline(path, wanted)
    length(rows) == length(wanted) || error("missing published boundary rows")
    return (
        lower_unstable=rows[6],
        lower_stable=rows[7],
        upper_stable=rows[11],
        upper_unstable=rows[12],
    )
end

function parse_screening_seeds(path::String)
    seeds = Dict{Tuple{String,String},NamedTuple}()
    for line in eachline(path)
        s = strip(line)
        isempty(s) && continue
        startswith(s,"#") && continue
        f = split(s,'\t')
        length(f) == 11 || error("invalid screening seed row: $line")
        key = (String(f[1]),String(f[2]))
        seeds[key] = (
            edge=String(f[1]), side=String(f[2]),
            m1=parse(BigFloat,f[3]), m2=parse(BigFloat,f[4]), m3=parse(BigFloat,f[5]),
            x1=parse(BigFloat,f[6]), v1=parse(BigFloat,f[7]), v2=parse(BigFloat,f[8]),
            period=parse(BigFloat,f[9]), screening_score=parse(BigFloat,f[10]),
            screening_residual=parse(BigFloat,f[11]),
        )
    end
    seeds
end

function chart_state(masses, p)
    m1,m2,m3 = masses
    x1,v1,v2,_ = p
    v3 = -(m1*v1+m2*v2)/m3
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
    stable = disc > 0 && abs(imag(t1)) <= sqrt(tol) && abs(imag(t2)) <= sqrt(tol) &&
             abs(t1) < 2 && abs(t2) < 2
    return (
        masses=masses,p=BigFloat.(p),closure=closure,M=M,sol=sol,
        alpha=alpha,beta=beta,disc=disc,t1=t1,t2=t2,score=score,stable=stable,
    )
end

function event_value(s, mode::String)
    mode == "plus_one" && return s.beta - 6s.alpha + 20
    mode == "minus_one" && return s.beta - 2s.alpha + 4
    mode == "trace_collision" && return s.disc
    error("unsupported event mode: $mode")
end

function infer_event_mode(a,b)
    crossings = Tuple{BigFloat,String}[]
    for mode in EVENT_MODES
        va,vb = event_value(a,mode),event_value(b,mode)
        if va == 0 || vb == 0 || sign(va) != sign(vb)
            push!(crossings,(max(abs(va),abs(vb)),mode))
        end
    end
    isempty(crossings) && error("no smooth Floquet event changes sign across published bracket")
    sort!(crossings,by=first)
    crossings[1][2]
end

function raw_row_sample(row; tol::BigFloat)
    masses=(row.m1,row.m2,row.m3)
    p=BigFloat[row.x1,row.v1,row.v2,row.period]
    orbit_sample(masses,p;tol=tol)
end

function raw_seed_sample(seed; tol::BigFloat)
    masses=(seed.m1,seed.m2,seed.m3)
    p=BigFloat[seed.x1,seed.v1,seed.v2,seed.period]
    orbit_sample(masses,p;tol=tol)
end

function validate_published_row(idx::Int,row; tol::BigFloat, max_raw_closure::BigFloat)
    s = raw_row_sample(row;tol=tol)
    expected = row.label == "S"
    println("baseline row=",idx," published=",row.label," computed=",s.stable,
            " raw_closure=",s.closure," score=",s.score)
    s.closure <= max_raw_closure || error(
        "published row $idx fails raw closure sanity gate: $(s.closure) > $max_raw_closure"
    )
    s.stable == expected || error(
        "published row $idx stability mismatch before correction: expected=$(row.label) score=$(s.score)"
    )
    s
end

function correct_chart(masses, p0; tol::BigFloat, target::BigFloat, maxiter::Int=12)
    p = BigFloat.(p0)
    last = nothing
    for iter in 1:maxiter
        s = orbit_sample(masses,p;tol=tol)
        residual = s.sol.u[end][1:8] - chart_state(masses,p)
        rn = norm(residual)
        last = (sample=s,residual=rn)
        println("  correction iter=",iter," closure=",rn)
        rn <= target && return p,s,iter

        C = chart_tangent(masses)
        J = zeros(BigFloat,8,4)
        J[:,1:3] .= (s.M-Matrix{BigFloat}(I,8,8))*C
        ffinal,_ = reduced_rhs_jacobian(s.sol.u[end][1:8],masses)
        J[:,4] .= ffinal
        # Generic rectangular least-squares uses QR and avoids normal equations.
        delta = J \ (-residual)
        p .+= delta
        p[4] > 0 || error("shooting produced non-positive period")
    end
    error("BigFloat shooting failed to reach closure target; last=$(last.residual)")
end

function corrected_sample(row;tol,target)
    masses=(row.m1,row.m2,row.m3)
    p,flow,iters = correct_chart(masses,[row.x1,row.v1,row.v2,row.period];tol=tol,target=target)
    merge(flow,(iterations=iters,))
end

function corrected_seed(seed;tol,target,max_raw_closure)
    raw = raw_seed_sample(seed;tol=tol)
    expected = seed.side == "stable"
    println("screening seed edge=",seed.edge," side=",seed.side,
            " raw_closure=",raw.closure," computed_stable=",raw.stable,
            " serialized_residual=",seed.screening_residual)
    raw.closure <= max_raw_closure || error(
        "screening seed fails independent raw closure sanity gate: $(raw.closure)"
    )
    p,flow,iters = correct_chart(raw.masses,raw.p;tol=tol,target=target)
    corrected = merge(flow,(iterations=iters,))
    corrected.stable == expected || println(
        "warning: screening side label changed after BigFloat correction edge=",seed.edge,
        " side=",seed.side," score=",corrected.score)
    corrected
end

function corrected_sample(m1,m2,m3,pguess;tol,target)
    masses=(m1,m2,m3)
    p,flow,iters = correct_chart(masses,pguess;tol=tol,target=target)
    merge(flow,(iterations=iters,))
end

function refine_event_edge(sa,sb,mode;tol,target,width,maxiter=80)
    va,vb = event_value(sa,mode),event_value(sb,mode)
    sign(va) == sign(vb) && error(
        "endpoint event values do not bracket zero for $mode: $va, $vb"
    )
    lo,hi = sa.masses[2] < sb.masses[2] ? (sa,sb) : (sb,sa)
    vlo,vhi = event_value(lo,mode),event_value(hi,mode)
    iterations=0
    while hi.masses[2]-lo.masses[2] > width && iterations < maxiter
        iterations += 1
        span=hi.masses[2]-lo.masses[2]
        # Safeguarded secant predictor; clipping prevents branch loss and ensures
        # the expensive BigFloat solve continues to shrink a valid bracket.
        denom=vhi-vlo
        secant = denom == 0 ? (lo.masses[2]+hi.masses[2])/2 : lo.masses[2]-vlo*span/denom
        guard=span/BigFloat(8)
        midm2=clamp(secant,lo.masses[2]+guard,hi.masses[2]-guard)
        theta=(midm2-lo.masses[2])/span
        pguess=(1-theta).*lo.p .+ theta.*hi.p
        mid=corrected_sample(lo.masses[1],midm2,lo.masses[3],pguess;tol=tol,target=target)
        vmid=event_value(mid,mode)
        println("boundary iter=",iterations," mode=",mode," m2=",midm2,
                " event=",vmid," closure=",mid.closure," width=",span)
        if vmid == 0
            lo=mid; hi=mid; vlo=vmid; vhi=vmid
            break
        elseif sign(vmid) == sign(vlo)
            lo=mid; vlo=vmid
        else
            hi=mid; vhi=vmid
        end
    end
    hi.masses[2]-lo.masses[2] <= width || error("boundary refinement did not reach width target")
    return lo,hi,iterations
end

function choose_seed_bracket(seeds,edge,mode;tol,target,max_raw_closure,fallback_a,fallback_b)
    try
        a=corrected_seed(seeds[(edge,"unstable")];tol=tol,target=target,max_raw_closure=max_raw_closure)
        b=corrected_seed(seeds[(edge,"stable")];tol=tol,target=target,max_raw_closure=max_raw_closure)
        va,vb=event_value(a,mode),event_value(b,mode)
        if va == 0 || vb == 0 || sign(va) != sign(vb)
            println("using independently validated narrow screening bracket for ",edge,
                    " mode=",mode," width=",abs(a.masses[2]-b.masses[2]))
            return a,b,"validated_screening_subgrid"
        end
        println("warning: corrected screening seeds no longer bracket ",mode,
                "; falling back to published grid")
    catch err
        println("warning: screening seed validation failed for ",edge,": ",err,
                "; falling back to published grid")
    end
    return corrected_sample(fallback_a;tol=tol,target=target),
           corrected_sample(fallback_b;tol=tol,target=target),
           "corrected_published_grid"
end

function sample_json(s,mode)
    "{" *
    "\"m1\":\"$(s.masses[1])\",\"m2\":\"$(s.masses[2])\",\"m3\":\"$(s.masses[3])\"," *
    "\"x1\":\"$(s.p[1])\",\"v1\":\"$(s.p[2])\",\"v2\":\"$(s.p[3])\",\"period\":\"$(s.p[4])\"," *
    "\"closure_norm\":\"$(s.closure)\",\"stability_score\":\"$(s.score)\"," *
    "\"alpha\":\"$(s.alpha)\",\"beta\":\"$(s.beta)\",\"discriminant\":\"$(s.disc)\"," *
    "\"trace_roots\":[$(json_complex(s.t1)),$(json_complex(s.t2))]," *
    "\"event_mode\":\"$mode\",\"event_value\":\"$(event_value(s,mode))\"" *
    "}"
end

function main_refine()
    length(ARGS) >= 2 || error("usage: refine_boundaries.jl DATASET OUTPUT [DPS] [TOL_EXP] [CLOSURE_EXP] [WIDTH_EXP] [SCREENING_SEEDS]")
    dataset,output=ARGS[1],ARGS[2]
    dps=length(ARGS)>=3 ? parse(Int,ARGS[3]) : 80
    tol_exp=length(ARGS)>=4 ? parse(Int,ARGS[4]) : 45
    closure_exp=length(ARGS)>=5 ? parse(Int,ARGS[5]) : 30
    width_exp=length(ARGS)>=6 ? parse(Int,ARGS[6]) : 12
    seed_path=length(ARGS)>=7 ? ARGS[7] : ""
    bits=ceil(Int,dps*log2(10))+32
    setprecision(BigFloat,bits) do
        tol=parse(BigFloat,"1e-$(tol_exp)")
        target=parse(BigFloat,"1e-$(closure_exp)")
        width=parse(BigFloat,"1e-$(width_exp)")
        max_raw_closure=BigFloat("1e-7")
        rows=published_brackets(dataset)

        # Frozen public rows are always the independent anchor and fail-fast gate.
        r6=validate_published_row(6,rows.lower_unstable;tol=tol,max_raw_closure=max_raw_closure)
        r7=validate_published_row(7,rows.lower_stable;tol=tol,max_raw_closure=max_raw_closure)
        r11=validate_published_row(11,rows.upper_stable;tol=tol,max_raw_closure=max_raw_closure)
        r12=validate_published_row(12,rows.upper_unstable;tol=tol,max_raw_closure=max_raw_closure)
        lower_mode=infer_event_mode(r6,r7)
        upper_mode=infer_event_mode(r11,r12)
        println("independent event modes lower=",lower_mode," upper=",upper_mode)

        seeds = isempty(seed_path) ? Dict{Tuple{String,String},NamedTuple}() : parse_screening_seeds(seed_path)
        lower_a,lower_b,lower_source = isempty(seeds) ?
            (corrected_sample(rows.lower_unstable;tol=tol,target=target), corrected_sample(rows.lower_stable;tol=tol,target=target), "corrected_published_grid") :
            choose_seed_bracket(seeds,"lower",lower_mode;tol=tol,target=target,max_raw_closure=max_raw_closure,
                fallback_a=rows.lower_unstable,fallback_b=rows.lower_stable)
        upper_a,upper_b,upper_source = isempty(seeds) ?
            (corrected_sample(rows.upper_stable;tol=tol,target=target), corrected_sample(rows.upper_unstable;tol=tol,target=target), "corrected_published_grid") :
            choose_seed_bracket(seeds,"upper",upper_mode;tol=tol,target=target,max_raw_closure=max_raw_closure,
                fallback_a=rows.upper_stable,fallback_b=rows.upper_unstable)

        lower_lo,lower_hi,lower_n=refine_event_edge(lower_a,lower_b,lower_mode;tol=tol,target=target,width=width)
        upper_lo,upper_hi,upper_n=refine_event_edge(upper_a,upper_b,upper_mode;tol=tol,target=target,width=width)
        mkpath(dirname(output))
        open(output,"w") do io
            print(io,"{\"implementation\":\"Julia BigFloat + Vern9 + variational QR shooting\",",
                "\"independent_anchor\":\"frozen Li-Li-Liao rows 6,7,11,12\",",
                "\"dps\":",dps,",\"ode_tolerance\":\"1e-",tol_exp,"\",",
                "\"closure_target\":\"1e-",closure_exp,"\",\"boundary_width_target\":\"1e-",width_exp,"\",",
                "\"lower\":{\"event_mode\":\"",lower_mode,"\",\"bracket_source\":\"",lower_source,
                "\",\"left\":",sample_json(lower_lo,lower_mode),",\"right\":",sample_json(lower_hi,lower_mode),",\"iterations\":",lower_n,"},",
                "\"upper\":{\"event_mode\":\"",upper_mode,"\",\"bracket_source\":\"",upper_source,
                "\",\"left\":",sample_json(upper_lo,upper_mode),",\"right\":",sample_json(upper_hi,upper_mode),",\"iterations\":",upper_n,"},",
                "\"claim_status\":\"independent high-precision boundary localization; canonical mechanism and full-manifold release gates still required\"}\n")
        end
    end
end

main_refine()
