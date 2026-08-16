#!/usr/bin/env julia

# Independent arbitrary-precision verification of the secondary G-=0 mass-plane
# projection fold.  The float64/JAX geometry workflow decides whether the two
# observed secondary -1 walls belong to one folded critical component.  This
# verifier supplies the independent equation-level fold test:
#
#     G-(m1,m2) = 0,
#     d G- / d m2 = 0,
#     d G- / d m1 != 0,
#     two independently corrected G-=0 roots at m1 > m1_fold straddle the root
#     with consistent nonzero secant curvature in the m1 projection.
#
# Here G- is evaluated only after independently correcting the periodic orbit at
# each mass pair with BigFloat Vern9/variational shooting.  Five-point mass
# stencils audit stationarity and transversality; the two catalog brackets audit
# the finite-scale fold curvature without differentiating through branch-switched
# periodic corrections.  No Python dynamics or derivatives enter.

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

function parse_branch_seeds(path::String)
    header=String[]
    rows=NamedTuple[]
    for line in eachline(path)
        s=strip(line)
        isempty(s) && continue
        startswith(s,"#") && continue
        f=split(s,'\t')
        if isempty(header)
            header=String.(f)
            continue
        end
        length(f)==length(header) || error("invalid branch seed row: $line")
        d=Dict(header[i]=>f[i] for i in eachindex(header))
        push!(rows,(
            name=String(d["name"]),
            cell_id=parse(Int,d["cell_id"]),
            orientation=String(d["orientation"]),
            m1=parse(BigFloat,d["m1"]),
            m2_lo=parse(BigFloat,d["m2_lo"]),
            m2_hi=parse(BigFloat,d["m2_hi"]),
            m2_seed=parse(BigFloat,d["m2_seed"]),
            m3=parse(BigFloat,d["m3"]),
            p=BigFloat[
                parse(BigFloat,d["x1"]),parse(BigFloat,d["v1"]),
                parse(BigFloat,d["v2"]),parse(BigFloat,d["period"]),
            ],
        ))
    end
    length(rows)==2 || error("expected exactly two secondary branch seeds")
    rows
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

function solve_branch_event(seed;tol,target,event_target,maxiter=8)
    current=corrected_at(seed.m1,seed.m2_seed,seed.m3,seed.p;tol=tol,target=target)
    derivative_h=min(parse(BigFloat,"2e-5"),(seed.m2_hi-seed.m2_lo)/8)
    for iter in 1:maxiter
        value=minus_event(current)
        println("branch iter=",iter," cell=",seed.cell_id," m2=",current.masses[2],
            " G-=",value," closure=",current.closure)
        if abs(value)<=event_target
            return current,iter
        end
        plus=corrected_at(seed.m1,current.masses[2]+derivative_h,seed.m3,current.p;
            tol=tol,target=target)
        minus=corrected_at(seed.m1,current.masses[2]-derivative_h,seed.m3,current.p;
            tol=tol,target=target)
        slope=(minus_event(plus)-minus_event(minus))/(2derivative_h)
        abs(slope)>=parse(BigFloat,"1e-3") || error("branch event derivative is singular")
        delta=-value/slope
        maxstep=(seed.m2_hi-seed.m2_lo)/4
        abs(delta)>maxstep && (delta=sign(delta)*maxstep)
        next_m2=clamp(current.masses[2]+delta,seed.m2_lo,seed.m2_hi)
        next_m2!=current.masses[2] || error("branch Newton stalled at bracket boundary")
        current=corrected_at(seed.m1,next_m2,seed.m3,current.p;tol=tol,target=target)
    end
    error("branch event solve did not reach target for cell $(seed.cell_id)")
end

function branch_json(seed,sample,iters,fold_m1,fold_m2)
    dm1=sample.masses[1]-fold_m1
    dm2=sample.masses[2]-fold_m2
    curvature=2dm1/(dm2*dm2)
    "{" *
    "\"name\":\"$(seed.name)\",\"cell_id\":$(seed.cell_id)," *
    "\"orientation\":\"$(seed.orientation)\"," *
    "\"source_m2_bracket\":[\"$(seed.m2_lo)\",\"$(seed.m2_hi)\"]," *
    "\"masses\":[\"$(sample.masses[1])\",\"$(sample.masses[2])\",\"$(sample.masses[3])\"]," *
    "\"chart\":[\"$(sample.p[1])\",\"$(sample.p[2])\",\"$(sample.p[3])\",\"$(sample.p[4])\"]," *
    "\"closure_norm\":\"$(sample.closure)\",\"minus_one_event\":\"$(minus_event(sample))\"," *
    "\"newton_iterations\":$iters,\"dm1_from_fold\":\"$dm1\"," *
    "\"dm2_from_fold\":\"$dm2\",\"secant_m1_curvature\":\"$curvature\"}"
end

function relative_change(a,b)
    abs(a-b)/max(abs(a),abs(b),parse(BigFloat,"1e-30"))
end

function audit_json(records)
    "["*join((
        "{\"h\":\"$(r.h)\",\"dGdm2\":\"$(r.dGdm2)\",\"d2Gdm22\":\"$(r.d2Gdm22)\",\"dGdm1\":\"$(r.dGdm1)\",\"m1_curvature\":\"$(r.m1_curvature)\"}"
        for r in records),",")*"]"
end

# Persist-and-SWAT: every expensive stage writes an INCOMPLETE checkpoint the
# moment it succeeds, so a job wall (which has already destroyed three runs of
# this verifier) can never again discard solved BigFloat work.  A checkpoint
# hardcodes "passed": false and carries no branch_curvature_audit, so
# scripts/classify_secondary_left_birth.py can never mistake one for the real
# artifact -- it requires bigfloat["passed"] === true plus a two-branch audit.
function write_checkpoint(path,stage,root,shift,iters,audit)
    mkpath(dirname(path))
    open(path,"w") do io
        print(io,
            "{\n",
            "  \"stage\": \"",stage,"\",\n",
            "  \"claim_status\": \"INCOMPLETE CHECKPOINT, NOT A RELEASE ARTIFACT. Written so a job wall never discards solved BigFloat work. Gates after this stage have not run.\",\n",
            "  \"masses\": [\"",root.masses[1],"\",\"",root.masses[2],"\",\"",root.masses[3],"\"],\n",
            "  \"chart\": [\"",root.p[1],"\",\"",root.p[2],"\",\"",root.p[3],"\",\"",root.p[4],"\"],\n",
            "  \"closure_norm\": \"",root.closure,"\",\n",
            "  \"minus_one_event\": \"",minus_event(root),"\",\n",
            "  \"mass_shift_from_screening_seed\": \"",shift,"\",\n",
            "  \"newton_iterations\": ",iters,",\n",
            "  \"stationarity_stencil_audit\": ",(audit===nothing ? "null" : audit_json(audit)),",\n",
            "  \"passed\": false\n",
            "}\n")
    end
    println("checkpoint written stage=",stage," -> ",path)
    flush(stdout)
end

function main()
    length(ARGS)>=3 || error("usage: verify_secondary_minus_fold.jl FOLD_SEED_TSV BRANCH_SEEDS_TSV OUTPUT [DPS] [TOL_EXP] [CLOSURE_EXP] [EVENT_EXP] [DERIV_EXP]")
    seed_path,branch_path,output=ARGS[1],ARGS[2],ARGS[3]
    dps=length(ARGS)>=4 ? parse(Int,ARGS[4]) : 60
    tol_exp=length(ARGS)>=5 ? parse(Int,ARGS[5]) : 28
    closure_exp=length(ARGS)>=6 ? parse(Int,ARGS[6]) : 18
    event_exp=length(ARGS)>=7 ? parse(Int,ARGS[7]) : 12
    deriv_exp=length(ARGS)>=8 ? parse(Int,ARGS[8]) : 8
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
        branch_seeds=parse_branch_seeds(branch_path)

        root,iters=solve_fold(seed;tol=tol,target=target,event_target=event_target,
            derivative_target=derivative_target,inner_h=inner_h,outer_h=outer_h,
            max_shift=max_shift,maxiter=8)
        root.closure<=target || error("fold root closure gate failed: $(root.closure)")
        abs(minus_event(root))<=event_target || error("fold G- gate failed: $(minus_event(root))")
        shift=norm(BigFloat[root.masses[1]-seed.m1,root.masses[2]-seed.m2])
        shift<=max_shift || error("fold root moved outside locality gate: $shift")
        write_checkpoint(output*".partial","fold_root",root,shift,iters,nothing)

        hs=BigFloat[parse(BigFloat,"8e-5"),parse(BigFloat,"4e-5"),parse(BigFloat,"2e-5"),parse(BigFloat,"1e-5")]
        audit=audit_fold(root;hs=hs,tol=tol,target=target)
        write_checkpoint(output*".partial","fold_root_and_stencil_audit",root,shift,iters,audit)
        last=audit[end]
        prev=audit[end-1]
        abs(last.dGdm2)<=parse(BigFloat,"1e-6") || error("fold stationarity failed at finest audit: $(last.dGdm2)")
        abs(last.dGdm2-prev.dGdm2)<=parse(BigFloat,"5e-6") || error("fold stationarity did not converge across stencils")
        abs(last.dGdm1)>=parse(BigFloat,"1") || error("fold transversality dG/dm1 too small: $(last.dGdm1)")
        relative_change(last.dGdm1,prev.dGdm1)<=parse(BigFloat,"0.05") || error("dG/dm1 stencil convergence gate failed")

        branch_results=NamedTuple[]
        for branch_seed in branch_seeds
            branch,branch_iters=solve_branch_event(branch_seed;tol=tol,target=target,
                event_target=event_target,maxiter=8)
            branch.closure<=target || error("branch closure gate failed: $(branch.closure)")
            abs(minus_event(branch))<=event_target || error("branch event gate failed")
            branch_seed.m2_lo<=branch.masses[2]<=branch_seed.m2_hi ||
                error("branch root left original source bracket")
            push!(branch_results,(seed=branch_seed,sample=branch,iters=branch_iters))
        end
        sort!(branch_results,by=x->x.sample.masses[2])
        lower,upper=branch_results
        lower.sample.masses[2]<root.masses[2]<upper.sample.masses[2] ||
            error("independent branch roots do not straddle the fold")
        curvatures=BigFloat[]
        for item in branch_results
            dm1=item.sample.masses[1]-root.masses[1]
            dm2=item.sample.masses[2]-root.masses[2]
            dm1>0 || error("newborn branch must lie above fold m1")
            push!(curvatures,2dm1/(dm2*dm2))
        end
        all(x->x>=parse(BigFloat,"0.1"),curvatures) ||
            error("two-sided secant curvature is too small: $curvatures")
        relative_change(curvatures[1],curvatures[2])<=parse(BigFloat,"0.15") ||
            error("two-sided secant curvatures disagree: $curvatures")

        mkpath(dirname(output))
        open(output,"w") do io
            print(io,
                "{\n",
                "  \"implementation\": \"independent Julia BigFloat Vern9 periodic correction + five-point mass derivatives\",\n",
                "  \"claim_status\": \"independent equation-level secondary minus-one fold verification anchored to both newborn source brackets; local reconnection geometry remains a separate gate\",\n",
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
                "  \"stationarity_stencil_audit\": ",audit_json(audit),",\n",
                "  \"branch_curvature_audit\": {\"method\":\"two independently corrected G-=0 roots at fixed m1, compared to the stationary fold root\",\"relative_curvature_disagreement\":\"",relative_change(curvatures[1],curvatures[2]),"\",\"branches\":[",
                branch_json(lower.seed,lower.sample,lower.iters,root.masses[1],root.masses[2]),",",
                branch_json(upper.seed,upper.sample,upper.iters,root.masses[1],root.masses[2]),"]},\n",
                "  \"passed\": true\n",
                "}\n")
        end
        println("secondary minus-one fold PASS masses=",root.masses[1],",",root.masses[2],
            " G-=",minus_event(root)," dGdm2=",last.dGdm2,
            " dGdm1=",last.dGdm1," secant_curvatures=",curvatures)
    end
end

main()
