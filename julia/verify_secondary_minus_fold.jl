#!/usr/bin/env julia

# Independent arbitrary-precision verification of the secondary G-=0 mass-plane
# projection fold.  The float64/JAX geometry workflow decides whether the two
# observed secondary -1 walls belong to one folded critical component.  This
# verifier supplies the independent equation-level nondegeneracy test:
#
#     G-(m1,m2) = 0,
#     d G- / d m2 = 0,
#     d G- / d m1 != 0,
#     d^2 G- / d m2^2 != 0.
#
# Here G- is evaluated only after independently correcting the periodic orbit at
# each mass pair with BigFloat Vern9/variational shooting.  A five-point mass
# stencil is used for the stationarity/curvature audit, and several stencil
# scales are frozen in the output.  No Python dynamics or derivatives enter.

include(joinpath(@__DIR__, "verify_critical_points.jl"))

function parse_fold_seed(path::String)
    header = String[]
    for line in eachline(path)
        s=strip(line)
        isempty(s) && continue
        startswith(s,"#") && continue
        f=split(s,'\t')
        if isempty(header)
            header=String.(f)
            continue
        end
        length(f)==length(header) || error("invalid fold seed row: $line")
        d=Dict(header[i]=>f[i] for i in eachindex(header))
        return (
            name=String(d["name"]),
            m1=parse(BigFloat,d["m1"]), m2=parse(BigFloat,d["m2"]), m3=parse(BigFloat,d["m3"]),
            p=BigFloat[parse(BigFloat,d["x1"]),parse(BigFloat,d["v1"]),parse(BigFloat,d["v2"]),parse(BigFloat,d["period"])],
        )
    end
    error("no fold seed parsed")
end

minus_event(s)=critical_event(s,"minus_one")

const CORRECTED_CACHE=Dict{Any,Any}()
const CORRECTED_CACHE_HITS=Ref(0)

function corrected_at(m1,m2,m3,pguess;tol,target)
    key=(m1,m2,m3,tol,target)
    if haskey(CORRECTED_CACHE,key)
        CORRECTED_CACHE_HITS[]+=1
        return CORRECTED_CACHE[key]
    end
    sample=corrected_sample(m1,m2,m3,pguess;tol=tol,target=target)
    CORRECTED_CACHE[key]=sample
    sample
end

function five_point_m2(center;h,tol,target)
    m1,m2,m3=center.masses
    p=center.p
    sm2=corrected_at(m1,m2-2h,m3,p;tol=tol,target=target)
    sm1=corrected_at(m1,m2-h,m3,p;tol=tol,target=target)
    sp1=corrected_at(m1,m2+h,m3,p;tol=tol,target=target)
    sp2=corrected_at(m1,m2+2h,m3,p;tol=tol,target=target)
    fm2,fm1,f0,fp1,fp2=minus_event(sm2),minus_event(sm1),minus_event(center),minus_event(sp1),minus_event(sp2)
    d1=(fm2-8fm1+8fp1-fp2)/(12h)
    d2=(-fp2+16fp1-30f0+16fm1-fm2)/(12h*h)
    return d1,d2,(sm2,sm1,sp1,sp2)
end

function five_point_m1(center;h,tol,target)
    m1,m2,m3=center.masses
    p=center.p
    sm2=corrected_at(m1-2h,m2,m3,p;tol=tol,target=target)
    sm1=corrected_at(m1-h,m2,m3,p;tol=tol,target=target)
    sp1=corrected_at(m1+h,m2,m3,p;tol=tol,target=target)
    sp2=corrected_at(m1+2h,m2,m3,p;tol=tol,target=target)
    (minus_event(sm2)-8minus_event(sm1)+8minus_event(sp1)-minus_event(sp2))/(12h)
end

function fold_vector(center;h,tol,target)
    d1,_,_=five_point_m2(center;h=h,tol=tol,target=target)
    BigFloat[minus_event(center),d1]
end

function singular_values_2x2(J)
    a,b,c,d=J[1,1],J[1,2],J[2,1],J[2,2]
    trjtj=a*a+b*b+c*c+d*d
    detj=a*d-b*c
    disc=max(zero(BigFloat),trjtj*trjtj-4detj*detj)
    root=sqrt(disc)
    BigFloat[sqrt(max(zero(BigFloat),(trjtj+root)/2)),sqrt(max(zero(BigFloat),(trjtj-root)/2))]
end

function fold_mass_jacobian(center;inner_h,outer_h,tol,target)
    m1,m2,m3=center.masses
    J=zeros(BigFloat,2,2)
    for j in 1:2
        if j==1
            plus=corrected_at(m1+outer_h,m2,m3,center.p;tol=tol,target=target)
            minus=corrected_at(m1-outer_h,m2,m3,center.p;tol=tol,target=target)
        else
            plus=corrected_at(m1,m2+outer_h,m3,center.p;tol=tol,target=target)
            minus=corrected_at(m1,m2-outer_h,m3,center.p;tol=tol,target=target)
        end
        J[:,j].=(fold_vector(plus;h=inner_h,tol=tol,target=target)-fold_vector(minus;h=inner_h,tol=tol,target=target))/(2outer_h)
    end
    J
end

function solve_fold(seed;tol,target,event_target,derivative_target,inner_h,outer_h,max_shift,maxiter=8)
    current=corrected_at(seed.m1,seed.m2,seed.m3,seed.p;tol=tol,target=target)
    seed_mass=BigFloat[seed.m1,seed.m2]
    for iter in 1:maxiter
        F=fold_vector(current;h=inner_h,tol=tol,target=target)
        println("fold iter=",iter," masses=",current.masses[1],",",current.masses[2],
                " G-=",F[1]," dGdm2=",F[2]," closure=",current.closure)
        if abs(F[1])<=event_target && abs(F[2])<=derivative_target
            return current,iter
        end
        J=fold_mass_jacobian(current;inner_h=inner_h,outer_h=outer_h,tol=tol,target=target)
        sv=singular_values_2x2(J)
        sv[end]>parse(BigFloat,"1e-20") || error("fold mass Jacobian is singular: sv=$sv")
        delta=J\(-F)
        stepnorm=norm(delta)
        maxstep=max_shift/2
        if stepnorm>maxstep
            delta .*= maxstep/stepnorm
        end
        accepted=nothing
        oldnorm=hypot(F[1]/event_target,F[2]/derivative_target)
        for ls in 0:6
            scale=BigFloat(2)^(-ls)
            trialmass=BigFloat[current.masses[1],current.masses[2]]+scale*delta
            norm(trialmass-seed_mass)<=max_shift || continue
            try
                trial=corrected_at(trialmass[1],trialmass[2],seed.m3,current.p;tol=tol,target=target)
                Ft=fold_vector(trial;h=inner_h,tol=tol,target=target)
                newnorm=hypot(Ft[1]/event_target,Ft[2]/derivative_target)
                println("  line search scale=",scale," scaled_norm=",newnorm)
                if newnorm<oldnorm
                    accepted=trial
                    break
                end
            catch err
                println("  trial correction failed scale=",scale," error=",err)
            end
        end
        accepted===nothing && error("fold Newton failed to reduce scaled residual; F=$F sv=$sv")
        current=accepted
    end
    error("fold solve did not reach event/stationarity targets")
end

function audit_fold(root;hs,tol,target)
    records=NamedTuple[]
    for h in hs
        d1,d2,_=five_point_m2(root;h=h,tol=tol,target=target)
        dm1=five_point_m1(root;h=h,tol=tol,target=target)
        curvature=-d2/dm1
        push!(records,(h=h,dGdm2=d1,d2Gdm22=d2,dGdm1=dm1,m1_curvature=curvature))
        println("audit h=",h," dGdm2=",d1," d2Gdm22=",d2," dGdm1=",dm1," m1''=",curvature)
    end
    records
end

function relative_change(a,b)
    abs(a-b)/max(abs(a),abs(b),parse(BigFloat,"1e-30"))
end

function audit_json(records)
    "["*join((
        "{\"h\":\"$(r.h)\",\"dGdm2\":\"$(r.dGdm2)\",\"d2Gdm22\":\"$(r.d2Gdm22)\",\"dGdm1\":\"$(r.dGdm1)\",\"m1_curvature\":\"$(r.m1_curvature)\"}"
        for r in records),",")*"]"
end

function main()
    length(ARGS)>=2 || error("usage: verify_secondary_minus_fold.jl SEED_TSV OUTPUT [DPS] [TOL_EXP] [CLOSURE_EXP] [EVENT_EXP] [DERIV_EXP]")
    seed_path,output=ARGS[1],ARGS[2]
    dps=length(ARGS)>=3 ? parse(Int,ARGS[3]) : 60
    tol_exp=length(ARGS)>=4 ? parse(Int,ARGS[4]) : 28
    closure_exp=length(ARGS)>=5 ? parse(Int,ARGS[5]) : 18
    event_exp=length(ARGS)>=6 ? parse(Int,ARGS[6]) : 12
    deriv_exp=length(ARGS)>=7 ? parse(Int,ARGS[7]) : 8
    bits=ceil(Int,dps*log2(10))+32

    setprecision(BigFloat,bits) do
        empty!(CORRECTED_CACHE)
        CORRECTED_CACHE_HITS[]=0
        tol=parse(BigFloat,"1e-$(tol_exp)")
        target=parse(BigFloat,"1e-$(closure_exp)")
        event_target=parse(BigFloat,"1e-$(event_exp)")
        derivative_target=parse(BigFloat,"1e-$(deriv_exp)")
        inner_h=parse(BigFloat,"2e-5")
        outer_h=parse(BigFloat,"2e-5")
        max_shift=parse(BigFloat,"1e-3")
        seed=parse_fold_seed(seed_path)

        root,iters=solve_fold(seed;tol=tol,target=target,event_target=event_target,
            derivative_target=derivative_target,inner_h=inner_h,outer_h=outer_h,
            max_shift=max_shift,maxiter=8)
        root.closure<=target || error("fold root closure gate failed: $(root.closure)")
        abs(minus_event(root))<=event_target || error("fold G- gate failed: $(minus_event(root))")
        shift=norm(BigFloat[root.masses[1]-seed.m1,root.masses[2]-seed.m2])
        shift<=max_shift || error("fold root moved outside locality gate: $shift")

        hs=BigFloat[parse(BigFloat,"8e-5"),parse(BigFloat,"4e-5"),parse(BigFloat,"2e-5"),parse(BigFloat,"1e-5")]
        audit=audit_fold(root;hs=hs,tol=tol,target=target)
        last=audit[end]
        prev=audit[end-1]
        abs(last.dGdm2)<=parse(BigFloat,"1e-6") || error("fold stationarity failed at finest audit: $(last.dGdm2)")
        abs(last.dGdm2-prev.dGdm2)<=parse(BigFloat,"5e-6") || error("fold stationarity did not converge across stencils")
        abs(last.dGdm1)>=parse(BigFloat,"1") || error("fold transversality dG/dm1 too small: $(last.dGdm1)")
        relative_change(last.dGdm1,prev.dGdm1)<=parse(BigFloat,"0.05") || error("dG/dm1 stencil convergence gate failed")
        abs(last.d2Gdm22)>=parse(BigFloat,"10") || error("fold curvature derivative too small: $(last.d2Gdm22)")
        relative_change(last.d2Gdm22,prev.d2Gdm22)<=parse(BigFloat,"0.10") || error("second-derivative stencil convergence gate failed")
        abs(last.m1_curvature)>=parse(BigFloat,"0.1") || error("projected m1 fold curvature too small: $(last.m1_curvature)")

        mkpath(dirname(output))
        open(output,"w") do io
            print(io,
                "{\n",
                "  \"implementation\": \"independent Julia BigFloat Vern9 periodic correction + five-point mass derivatives\",\n",
                "  \"claim_status\": \"independent equation-level secondary minus-one fold/nondegeneracy verification; branch reconnection geometry remains a separate gate\",\n",
                "  \"name\": \"",seed.name,"\",\n",
                "  \"dps\": ",dps,",\n",
                "  \"ode_tolerance\": \"1e-",tol_exp,"\",\n",
                "  \"closure_target\": \"1e-",closure_exp,"\",\n",
                "  \"event_target\": \"1e-",event_exp,"\",\n",
                "  \"stationarity_solve_target\": \"1e-",deriv_exp,"\",\n",
                "  \"masses\": [\"",root.masses[1],"\",\"",root.masses[2],"\",\"",root.masses[3],"\"],\n",
                "  \"chart\": [\"",root.p[1],"\",\"",root.p[2],"\",\"",root.p[3],"\",\"",root.p[4],"\"],\n",
                "  \"closure_norm\": \"",root.closure,"\",\n",
                "  \"minus_one_event\": \"",minus_event(root),"\",\n",
                "  \"alpha\": \"",root.alpha,"\",\n",
                "  \"beta\": \"",root.beta,"\",\n",
                "  \"discriminant\": \"",root.disc,"\",\n",
                "  \"mass_shift_from_screening_seed\": \"",shift,"\",\n",
                "  \"newton_iterations\": ",iters,",\n",
                "  \"correction_cache_entries\": ",length(CORRECTED_CACHE),",\n",
                "  \"correction_cache_hits\": ",CORRECTED_CACHE_HITS[],",\n",
                "  \"stencil_audit\": ",audit_json(audit),",\n",
                "  \"passed\": true\n",
                "}\n")
        end
        println("secondary minus-one fold PASS masses=",root.masses[1],",",root.masses[2],
            " G-=",minus_event(root)," dGdm2=",last.dGdm2," d2=",last.d2Gdm22,
            " dGdm1=",last.dGdm1," m1_curvature=",last.m1_curvature)
    end
end

main()
