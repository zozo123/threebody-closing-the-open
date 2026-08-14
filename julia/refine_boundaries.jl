#!/usr/bin/env julia

include(joinpath(@__DIR__, "verify_reduced.jl"))

function parse_seeds(path::String)
    seeds = Dict{Tuple{String,String},NamedTuple}()
    for line in eachline(path)
        s = strip(line)
        isempty(s) && continue
        startswith(s,"#") && continue
        f = split(s,'\t')
        length(f) == 11 || error("invalid seed row: $line")
        key = (f[1],f[2])
        seeds[key] = (
            edge=f[1], side=f[2],
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

function correct_chart(masses, p0; tol::BigFloat, target::BigFloat, maxiter::Int=12)
    p = BigFloat.(p0)
    last = nothing
    for iter in 1:maxiter
        z0 = chart_state(masses,p)
        zf,M,sol = integrate_augmented(z0,masses,p[4];tol=tol)
        residual = zf-z0
        rn = norm(residual)
        last = (zf=zf,M=M,sol=sol,residual=rn)
        println("  correction iter=",iter," closure=",rn)
        rn <= target && return p,last,iter
        C = chart_tangent(masses)
        J = zeros(BigFloat,8,4)
        J[:,1:3] .= (M-Matrix{BigFloat}(I,8,8))*C
        ffinal,_ = reduced_rhs_jacobian(zf,masses)
        J[:,4] .= ffinal
        delta = -(transpose(J)*J) \ (transpose(J)*residual)
        p .+= delta
        p[4] > 0 || error("shooting produced non-positive period")
    end
    error("BigFloat shooting failed to reach closure target; last=$(last.residual)")
end

function corrected_sample(m1,m2,m3,pguess;tol,target)
    masses=(m1,m2,m3)
    p,flow,iters = correct_chart(masses,pguess;tol=tol,target=target)
    alpha,beta,disc,t1,t2,score = monodromy_invariants(flow.M)
    return (
        masses=masses,p=p,closure=flow.residual,iterations=iters,
        alpha=alpha,beta=beta,disc=disc,t1=t1,t2=t2,score=score,
    )
end

function refine_edge(a,b;tol,target,width,maxiter=60)
    # Endpoints may arrive in either parameter order. Their score signs must differ.
    sa = corrected_sample(a.m1,a.m2,a.m3,[a.x1,a.v1,a.v2,a.period];tol=tol,target=target)
    sb = corrected_sample(b.m1,b.m2,b.m3,[b.x1,b.v1,b.v2,b.period];tol=tol,target=target)
    sign(sa.score) == sign(sb.score) && error("endpoint stability scores do not bracket zero")
    lo,hi = sa.masses[2] < sb.masses[2] ? (sa,sb) : (sb,sa)
    iterations=0
    while hi.masses[2]-lo.masses[2] > width && iterations < maxiter
        iterations += 1
        midm2=(lo.masses[2]+hi.masses[2])/2
        # Linear chart interpolation is a better predictor than selecting one side.
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
    return lo,hi,iterations
end

function sample_json(s)
    "{" *
    "\"m1\":\"$(s.masses[1])\",\"m2\":\"$(s.masses[2])\",\"m3\":\"$(s.masses[3])\"," *
    "\"x1\":\"$(s.p[1])\",\"v1\":\"$(s.p[2])\",\"v2\":\"$(s.p[3])\",\"period\":\"$(s.p[4])\"," *
    "\"closure_norm\":\"$(s.closure)\",\"stability_score\":\"$(s.score)\"," *
    "\"alpha\":\"$(s.alpha)\",\"beta\":\"$(s.beta)\",\"discriminant\":\"$(s.disc)\"," *
    "\"trace_roots\":[$(json_complex(s.t1)),$(json_complex(s.t2))]" *
    "}"
end

function main_refine()
    length(ARGS) >= 2 || error("usage: refine_boundaries.jl SEEDS OUTPUT [DPS] [TOL_EXP] [CLOSURE_EXP] [WIDTH_EXP]")
    seeds_path,output=ARGS[1],ARGS[2]
    dps=length(ARGS)>=3 ? parse(Int,ARGS[3]) : 80
    tol_exp=length(ARGS)>=4 ? parse(Int,ARGS[4]) : 45
    closure_exp=length(ARGS)>=5 ? parse(Int,ARGS[5]) : 35
    width_exp=length(ARGS)>=6 ? parse(Int,ARGS[6]) : 12
    bits=ceil(Int,dps*log2(10))+32
    setprecision(BigFloat,bits) do
        tol=parse(BigFloat,"1e-$(tol_exp)")
        target=parse(BigFloat,"1e-$(closure_exp)")
        width=parse(BigFloat,"1e-$(width_exp)")
        seeds=parse_seeds(seeds_path)
        lower_lo,lower_hi,lower_n=refine_edge(seeds[("lower","unstable")],seeds[("lower","stable")];tol=tol,target=target,width=width)
        upper_lo,upper_hi,upper_n=refine_edge(seeds[("upper","stable")],seeds[("upper","unstable")];tol=tol,target=target,width=width)
        mkpath(dirname(output))
        open(output,"w") do io
            print(io,"{\"implementation\":\"Julia BigFloat + Vern9 + variational Gauss-Newton\",",
                "\"dps\":",dps,",\"ode_tolerance\":\"1e-",tol_exp,"\",",
                "\"closure_target\":\"1e-",closure_exp,"\",\"boundary_width_target\":\"1e-",width_exp,"\",",
                "\"lower\":{\"left\":",sample_json(lower_lo),",\"right\":",sample_json(lower_hi),",\"iterations\":",lower_n,"},",
                "\"upper\":{\"left\":",sample_json(upper_lo),",\"right\":",sample_json(upper_hi),",\"iterations\":",upper_n,"},",
                "\"claim_status\":\"independent high-precision computation; novelty and second-implementation release gate still required\"}\n")
        end
    end
end

main_refine()
