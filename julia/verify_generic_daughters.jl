#!/usr/bin/env julia

# Independent arbitrary-precision verification of generic lower-+1 daughter
# periodic orbits.  The Python generic corrector is not imported.  Frozen
# float64 trace points provide only initial state/period seeds.
#
# Unknowns are (z0[1:8], T) at fixed masses.  We solve the full eight periodic
# closure equations together with scale, rotation and phase gauges as an
# overdetermined 11x9 BigFloat Gauss-Newton problem using the independently
# integrated variational monodromy.  The resulting generic orbit is then
# compared with an independently BigFloat-corrected Li parent at the same mass.

include(joinpath(@__DIR__, "verify_critical_points.jl"))

function parse_generic_daughter_seeds(path::String)
    seeds = NamedTuple[]
    header = String[]
    for line in eachline(path)
        s = strip(line)
        isempty(s) && continue
        startswith(s, "#") && continue
        f = split(s, '\t')
        if isempty(header)
            header = String.(f)
            continue
        end
        length(f) == length(header) || error("invalid daughter seed row: $line")
        d = Dict(header[i] => f[i] for i in eachindex(header))
        z = BigFloat[parse(BigFloat,d["z$(i)"]) for i in 1:8]
        push!(seeds, (
            name=String(d["name"]),
            masses=(parse(BigFloat,d["m1"]),parse(BigFloat,d["m2"]),parse(BigFloat,d["m3"])),
            z=z,
            period=parse(BigFloat,d["period"]),
        ))
    end
    isempty(seeds) && error("no generic daughter seeds parsed")
    seeds
end

function generic_residual_jacobian(masses, y, reference; tol::BigFloat)
    z0 = BigFloat.(y[1:8])
    period = BigFloat(y[9])
    period > 0 || error("generic daughter correction produced non-positive period")
    zf,M,sol = integrate_augmented(z0,masses,period;tol=tol)
    closure = zf-z0

    ffinal,_ = reduced_rhs_jacobian(zf,masses)
    fref,_ = reduced_rhs_jacobian(reference,masses)
    phase_scale = max(norm(fref),BigFloat(1))
    scale = z0[3]^2 + z0[4]^2 - 1
    rotation = z0[4]
    phase = dot(z0-reference,fref)/phase_scale
    residual = vcat(closure,BigFloat[scale,rotation,phase])

    J = zeros(BigFloat,11,9)
    J[1:8,1:8] .= M-Matrix{BigFloat}(I,8,8)
    J[1:8,9] .= ffinal
    J[9,3] = 2z0[3]
    J[9,4] = 2z0[4]
    J[10,4] = 1
    J[11,1:8] .= fref/phase_scale
    return residual,J,(;zf,M,sol,closure,scale,rotation,phase)
end

function correct_generic_daughter(seed;tol,target,maxiter=12)
    masses=seed.masses
    reference=copy(seed.z)
    y=vcat(copy(seed.z),seed.period)
    last=BigFloat(Inf)
    for iter in 1:maxiter
        residual,J,data=generic_residual_jacobian(masses,y,reference;tol=tol)
        rn=norm(residual)
        last=rn
        println("generic correction iter=",iter," name=",seed.name," residual=",rn,
                " closure=",norm(data.closure)," gauges=",norm(residual[9:11]))
        if rn <= target
            return y,data,iter
        end
        delta=J\(-residual)
        accepted=false
        for ls in 0:7
            λ=BigFloat(2)^(-ls)
            trial=y+λ*delta
            trial[9] > 0 || continue
            tr,_,_=generic_residual_jacobian(masses,trial,reference;tol=tol)
            if norm(tr) < rn
                y=trial
                accepted=true
                break
            end
        end
        accepted || error("generic daughter line search failed at residual=$rn")
    end
    error("BigFloat generic daughter correction failed; last residual=$last")
end

function scaled_parent_distance(z,period,parent_state,parent_period)
    floors=BigFloat[0.2,0.2,0.5,0.2,0.5,0.5,0.5,0.5,1.0]
    a=vcat(z,period)
    b=vcat(parent_state,parent_period)
    scales=max.(max.(abs.(a),abs.(b)),floors)
    norm((a-b)./scales)
end

function off_li_norm(z)
    norm(BigFloat[z[2],z[5],z[7]])
end

function daughter_json(seed,y,data,iters,parent,parent_state,parent_distance)
    z=y[1:8]; period=y[9]
    closure=norm(data.closure)
    gauges=norm(BigFloat[data.scale,data.rotation,data.phase])
    "{" *
      "\"name\":\"$(seed.name)\"," *
      "\"masses\":[\"$(seed.masses[1])\",\"$(seed.masses[2])\",\"$(seed.masses[3])\"]," *
      "\"state\":[" * join(("\"$(x)\"" for x in z),",") * "]," *
      "\"period\":\"$period\"," *
      "\"closure_norm\":\"$closure\",\"gauge_norm\":\"$gauges\"," *
      "\"off_li_norm\":\"$(off_li_norm(z))\"," *
      "\"parent_distance\":\"$parent_distance\"," *
      "\"parent_closure_norm\":\"$(parent.closure)\"," *
      "\"parent_state\":[" * join(("\"$(x)\"" for x in parent_state),",") * "]," *
      "\"parent_period\":\"$(parent.p[4])\",\"iterations\":$iters" *
      "}"
end

function main_generic_daughters()
    length(ARGS)>=2 || error("usage: verify_generic_daughters.jl SEEDS OUTPUT [DPS] [TOL_EXP] [TARGET_EXP] [MIN_PARENT_DISTANCE_EXP]")
    seed_path,output=ARGS[1],ARGS[2]
    dps=length(ARGS)>=3 ? parse(Int,ARGS[3]) : 50
    tol_exp=length(ARGS)>=4 ? parse(Int,ARGS[4]) : 25
    target_exp=length(ARGS)>=5 ? parse(Int,ARGS[5]) : 18
    min_parent_exp=length(ARGS)>=6 ? parse(Int,ARGS[6]) : 4
    bits=ceil(Int,dps*log2(10))+32

    setprecision(BigFloat,bits) do
        tol=parse(BigFloat,"1e-$(tol_exp)")
        target=parse(BigFloat,"1e-$(target_exp)")
        min_parent_distance=parse(BigFloat,"1e-$(min_parent_exp)")
        seeds=parse_generic_daughter_seeds(seed_path)
        results=String[]

        # Same frozen lower-+1 parent chart seed used only to initialize an
        # independent fixed-mass BigFloat Li correction at each daughter mass.
        parent_guess=BigFloat[
            parse(BigFloat,"-0.13560639235459448"),
            parse(BigFloat,"2.506249952852668"),
            parse(BigFloat,"0.3183937222016764"),
            parse(BigFloat,"5.166023712927732"),
        ]

        for seed in seeds
            println("verifying generic daughter ",seed.name)
            y,data,iters=correct_generic_daughter(seed;tol=tol,target=target)
            closure=norm(data.closure)
            gauges=norm(BigFloat[data.scale,data.rotation,data.phase])
            closure <= target || error("daughter closure gate failed for $(seed.name): $closure")
            gauges <= target || error("daughter gauge gate failed for $(seed.name): $gauges")

            parent=corrected_sample(
                seed.masses[1],seed.masses[2],seed.masses[3],parent_guess;
                tol=tol,target=target,
            )
            parent_state=chart_state(parent.masses,parent.p)
            pd=scaled_parent_distance(y[1:8],y[9],parent_state,parent.p[4])
            off=off_li_norm(y[1:8])
            println("  parent distance=",pd," off_li=",off," parent closure=",parent.closure)
            pd >= min_parent_distance || error("daughter collapsed onto Li parent: distance=$pd")
            off >= min_parent_distance/10 || error("daughter symmetry-breaking norm too small: $off")
            push!(results,daughter_json(seed,y,data,iters,parent,parent_state,pd))
        end

        mkpath(dirname(output))
        open(output,"w") do io
            print(io,
              "{\"implementation\":\"independent Julia BigFloat + Vern9 + full 8D variational generic Gauss-Newton\",",
              "\"dps\":",dps,",\"ode_tolerance\":\"1e-",tol_exp,"\",",
              "\"residual_target\":\"1e-",target_exp,"\",",
              "\"minimum_parent_distance\":\"1e-",min_parent_exp,"\",",
              "\"results\":[",join(results,","),"],",
              "\"claim_status\":\"independent BigFloat reproduction of generic daughter periodic orbits and separation from the independently corrected Li parent; global daughter genealogy remains a separate continuation gate\"}\n"
            )
        end
    end
end

if abspath(PROGRAM_FILE)==@__FILE__
    main_generic_daughters()
end
