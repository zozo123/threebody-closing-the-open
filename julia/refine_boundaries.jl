#!/usr/bin/env julia

include(joinpath(@__DIR__, "verify_reduced.jl"))

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

function raw_row_sample(row; tol::BigFloat)
    masses=(row.m1,row.m2,row.m3)
    p=BigFloat[row.x1,row.v1,row.v2,row.period]
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

        # Rectangular least-squares solve uses QR rather than normal equations.
        # This avoids squaring the shooting Jacobian condition number.
        delta = J \ (-residual)
        p .+= delta
        p[4] > 0 || error("shooting produced non-positive period")
    end
    error("BigFloat shooting failed to reach closure target; last=$(last.residual)")
end

function corrected_sample(row;tol,target)
    masses=(row.m1,row.m2,row.m3)
    p,flow,iters = correct_chart(masses,[row.x1,row.v1,row.v2,row.period];tol=tol,target=target)
    return merge(flow,(iterations=iters,))
end

function corrected_sample(m1,m2,m3,pguess;tol,target)
    masses=(m1,m2,m3)
    p,flow,iters = correct_chart(masses,pguess;tol=tol,target=target)
    return merge(flow,(iterations=iters,))
end

function refine_edge(a,b;tol,target,width,maxiter=80)
    sa = corrected_sample(a;tol=tol,target=target)
    sb = corrected_sample(b;tol=tol,target=target)
    sign(sa.score) == sign(sb.score) && error(
        "corrected published endpoints do not bracket zero: scores=$(sa.score), $(sb.score)"
    )
    lo,hi = sa.masses[2] < sb.masses[2] ? (sa,sb) : (sb,sa)
    iterations=0
    while hi.masses[2]-lo.masses[2] > width && iterations < maxiter
        iterations += 1
        midm2=(lo.masses[2]+hi.masses[2])/2
        theta=(midm2-lo.masses[2])/(hi.masses[2]-lo.masses[2])
        pguess=(1-theta).*lo.p .+ theta.*hi.p
        mid=corrected_sample(lo.masses[1],midm2,lo.masses[3],pguess;tol=tol,target=target)
        println("boundary iter=",iterations," m2=",midm2," score=",mid.score,
                " closure=",mid.closure)
        if sign(mid.score) == sign(lo.score)
            lo=mid
        else
            hi=mid
        end
    end
    hi.masses[2]-lo.masses[2] <= width || error("boundary refinement did not reach width target")
    return lo,hi,iterations
end

function active_constraint(s)
    vals = (s.disc, BigFloat(2)-abs(s.t1), BigFloat(2)-abs(s.t2))
    labels = ("discriminant", "trace_root_1_abs_2", "trace_root_2_abs_2")
    labels[argmin(vals)]
end

function sample_json(s)
    "{" *
    "\"m1\":\"$(s.masses[1])\",\"m2\":\"$(s.masses[2])\",\"m3\":\"$(s.masses[3])\"," *
    "\"x1\":\"$(s.p[1])\",\"v1\":\"$(s.p[2])\",\"v2\":\"$(s.p[3])\",\"period\":\"$(s.p[4])\"," *
    "\"closure_norm\":\"$(s.closure)\",\"stability_score\":\"$(s.score)\"," *
    "\"alpha\":\"$(s.alpha)\",\"beta\":\"$(s.beta)\",\"discriminant\":\"$(s.disc)\"," *
    "\"trace_roots\":[$(json_complex(s.t1)),$(json_complex(s.t2))]," *
    "\"active_constraint\":\"$(active_constraint(s))\"" *
    "}"
end

function main_refine()
    length(ARGS) >= 2 || error("usage: refine_boundaries.jl DATASET OUTPUT [DPS] [TOL_EXP] [CLOSURE_EXP] [WIDTH_EXP]")
    dataset,output=ARGS[1],ARGS[2]
    dps=length(ARGS)>=3 ? parse(Int,ARGS[3]) : 80
    tol_exp=length(ARGS)>=4 ? parse(Int,ARGS[4]) : 45
    closure_exp=length(ARGS)>=5 ? parse(Int,ARGS[5]) : 30
    width_exp=length(ARGS)>=6 ? parse(Int,ARGS[6]) : 12
    bits=ceil(Int,dps*log2(10))+32
    setprecision(BigFloat,bits) do
        tol=parse(BigFloat,"1e-$(tol_exp)")
        target=parse(BigFloat,"1e-$(closure_exp)")
        width=parse(BigFloat,"1e-$(width_exp)")
        max_raw_closure=BigFloat("1e-7")
        rows=published_brackets(dataset)

        # Fail fast if the frozen public data are not independently reproduced.
        validate_published_row(6,rows.lower_unstable;tol=tol,max_raw_closure=max_raw_closure)
        validate_published_row(7,rows.lower_stable;tol=tol,max_raw_closure=max_raw_closure)
        validate_published_row(11,rows.upper_stable;tol=tol,max_raw_closure=max_raw_closure)
        validate_published_row(12,rows.upper_unstable;tol=tol,max_raw_closure=max_raw_closure)

        lower_lo,lower_hi,lower_n=refine_edge(rows.lower_unstable,rows.lower_stable;tol=tol,target=target,width=width)
        upper_lo,upper_hi,upper_n=refine_edge(rows.upper_stable,rows.upper_unstable;tol=tol,target=target,width=width)
        mkpath(dirname(output))
        open(output,"w") do io
            print(io,"{\"implementation\":\"Julia BigFloat + Vern9 + variational QR shooting\",",
                "\"seed_source\":\"frozen Li-Li-Liao rows 6,7,11,12\",",
                "\"dps\":",dps,",\"ode_tolerance\":\"1e-",tol_exp,"\",",
                "\"closure_target\":\"1e-",closure_exp,"\",\"boundary_width_target\":\"1e-",width_exp,"\",",
                "\"lower\":{\"left\":",sample_json(lower_lo),",\"right\":",sample_json(lower_hi),",\"iterations\":",lower_n,"},",
                "\"upper\":{\"left\":",sample_json(upper_lo),",\"right\":",sample_json(upper_hi),",\"iterations\":",upper_n,"},",
                "\"claim_status\":\"independent high-precision boundary localization; canonical mechanism and full-manifold release gates still required\"}\n")
        end
    end
end

main_refine()
