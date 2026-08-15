#!/usr/bin/env julia

# Latest-stack independent cross-check for one frozen mixed (+1,-1) organizer.
#
# This file intentionally does not include/import the frozen Julia verifier or
# any Python implementation. It reconstructs the reduced dynamics, variational
# equations, Li normalization and mixed Floquet events from first principles,
# then lets BifurcationKit solve a rank-revealed six-equation organizer system.
# This is an independent Float64 cross-check, NOT the arbitrary-precision release gate.

using BifurcationKit
const BK = BifurcationKit
using LinearAlgebra
using OrdinaryDiffEq
using OrdinaryDiffEqVerner
using SciMLBase
using Pkg

function read_seed(path::String)
    lines = [strip(x) for x in readlines(path) if !isempty(strip(x)) && !startswith(strip(x), "#")]
    length(lines) >= 2 || error("seed TSV needs header plus one row")
    header = split(lines[1], '\t')
    vals = split(lines[2], '\t')
    length(header) == length(vals) || error("seed TSV column mismatch")
    d = Dict(header[i] => vals[i] for i in eachindex(header))
    return (
        name = d["name"],
        m1 = parse(Float64,d["m1"]), m2 = parse(Float64,d["m2"]), m3 = parse(Float64,d["m3"]),
        x1 = parse(Float64,d["x1"]), v1 = parse(Float64,d["v1"]), v2 = parse(Float64,d["v2"]),
        period = parse(Float64,d["period"]),
    )
end

@inline function force2(x::T, y::T) where {T<:Real}
    r2 = x*x + y*y
    r2 == 0 && error("binary collision")
    r = sqrt(r2)
    inv3 = inv(r*r*r)
    return x*inv3, y*inv3
end

function reduced!(du, z, p, t)
    m1,m2,m3 = p.m1,p.m2,p.m3
    q1x,q1y,q2x,q2y = z[1],z[2],z[3],z[4]
    d12x,d12y = q2x-q1x,q2y-q1y
    f1x,f1y = force2(q1x,q1y)
    f2x,f2y = force2(q2x,q2y)
    fdx,fdy = force2(d12x,d12y)
    du[1] = z[5]; du[2] = z[6]; du[3] = z[7]; du[4] = z[8]
    du[5] = m2*fdx - (m1+m3)*f1x - m2*f2x
    du[6] = m2*fdy - (m1+m3)*f1y - m2*f2y
    du[7] = -m1*fdx - m1*f1x - (m2+m3)*f2x
    du[8] = -m1*fdy - m1*f1y - (m2+m3)*f2y
    return nothing
end

function g_and_dg(x::AbstractVector{T}) where {T<:Real}
    r2 = x[1]^2 + x[2]^2
    r2 == 0 && error("binary collision")
    r = sqrt(r2)
    inv3 = inv(r^3); inv5 = inv(r^5)
    g = T[x[1]*inv3,x[2]*inv3]
    d = T[inv3-3*x[1]^2*inv5 -3*x[1]*x[2]*inv5;
          -3*x[2]*x[1]*inv5 inv3-3*x[2]^2*inv5]
    return g,d
end

function reduced_jacobian(z, p)
    T = eltype(z)
    m1,m2,m3 = p.m1,p.m2,p.m3
    q1 = z[1:2]; q2 = z[3:4]; d12 = q2-q1
    _,d1 = g_and_dg(q1); _,d2 = g_and_dg(q2); _,dd = g_and_dg(d12)
    J = zeros(T,8,8)
    J[1:4,5:8] .= Matrix{T}(I,4,4)
    J[5:6,1:2] .= -m2.*dd .- (m1+m3).*d1
    J[5:6,3:4] .=  m2.*dd .- m2.*d2
    J[7:8,1:2] .=  m1.*dd .- m1.*d1
    J[7:8,3:4] .= -m1.*dd .- (m2+m3).*d2
    return J
end

function chart_state(q, p)
    x1,v1,v2,T = q
    v3 = -(p.m1*v1 + p.m2*v2)/p.m3
    return [x1,0.0,1.0,0.0,0.0,v1-v3,0.0,v2-v3]
end

function chart_tangent(p)
    C = zeros(Float64,8,3)
    C[1,1] = 1.0
    C[6,2] = 1 + p.m1/p.m3
    C[8,2] = p.m1/p.m3
    C[6,3] = p.m2/p.m3
    C[8,3] = 1 + p.m2/p.m3
    return C
end

function augmented!(du,u,p,t)
    z = @view u[1:8]
    reduced!(@view(du[1:8]),z,p,t)
    Phi = reshape(@view(u[9:72]),8,8)
    dPhi = reduced_jacobian(z,p)*Phi
    du[9:72] .= vec(dPhi)
    return nothing
end

function flow_data(q,p; rtol=2e-11, atol=2e-13)
    Tper = q[4]
    Tper > 0 || error("non-positive period")
    z0 = chart_state(q,p)
    u0 = vcat(z0,vec(Matrix{Float64}(I,8,8)))
    prob = ODEProblem(augmented!,u0,(0.0,Tper),p)
    sol = solve(prob,Vern9();reltol=rtol,abstol=atol,save_everystep=false,maxiters=100_000_000)
    SciMLBase.successful_retcode(sol) || error("Vern9 integration failed: $(sol.retcode)")
    uf = sol.u[end]
    zf = Vector(uf[1:8]); M = reshape(Vector(uf[9:72]),8,8)
    closure = zf-z0
    C = chart_tangent(p)
    Jshoot = zeros(Float64,8,4)
    Jshoot[:,1:3] .= (M-I)*C
    dz = similar(zf); reduced!(dz,zf,p,Tper)
    Jshoot[:,4] .= dz
    alpha = tr(M)
    beta = (alpha^2-tr(M*M))/2
    gp = beta-6alpha+20
    gm = beta-2alpha+4
    disc = (alpha-4)^2 - 4*(beta-4alpha+8)
    return (;closure,M,Jshoot,alpha,beta,gp,gm,disc)
end

function best_rows(J)
    best = (1,2,3,4); bestdet = -Inf
    for i in 1:5, j in i+1:6, k in j+1:7, l in k+1:8
        rows = (i,j,k,l)
        d = abs(det(J[collect(rows),:]))
        if isfinite(d) && d > bestdet
            bestdet = d; best = rows
        end
    end
    bestdet > 1e-14 || error("rank-revealing closure row selection failed: det=$bestdet")
    return collect(best),bestdet
end

function package_version(name)
    for (_,dep) in Pkg.dependencies()
        dep.name == name && return string(dep.version)
    end
    return "unknown"
end

function json_string(s)
    replace(string(s), "\\"=>"\\\\", "\""=>"\\\"")
end

function main()
    length(ARGS) == 2 || error("usage: crosscheck_mixed_vertex.jl SEED_TSV OUTPUT_JSON")
    seed = read_seed(ARGS[1]); output = ARGS[2]
    p0 = (m1=seed.m1,m2=seed.m2,m3=seed.m3)
    q0 = [seed.x1,seed.v1,seed.v2,seed.period]
    initial = flow_data(q0,p0)
    rows,rowdet = best_rows(initial.Jshoot)

    closure_scale = 1e-5
    event_scale = 1e-4
    cache_key = Ref{Any}(nothing); cache_value = Ref{Any}(nothing)
    function evaluate(y)
        key = Tuple(Float64.(y))
        if cache_key[] != key
            p = (m1=Float64(y[5]),m2=Float64(y[6]),m3=seed.m3)
            q = Float64.(y[1:4])
            cache_key[] = key
            cache_value[] = flow_data(q,p;rtol=3e-11,atol=3e-13)
        end
        return cache_value[]
    end
    function residual6(y, dummy)
        d = evaluate(y)
        return vcat(d.closure[rows]./closure_scale,[d.gp/event_scale,d.gm/event_scale])
    end
    function jac6(y, dummy)
        y0 = Float64.(y); J = zeros(Float64,6,6)
        for j in 1:6
            h = 1e-6*max(abs(y0[j]),1.0)
            yp=copy(y0); ym=copy(y0); yp[j]+=h; ym[j]-=h
            J[:,j] .= (residual6(yp,dummy)-residual6(ym,dummy))/(2h)
        end
        return J
    end

    y0 = [q0... ,seed.m1,seed.m2]
    prob = BifurcationProblem(residual6,y0,(dummy=0.0,),(@optic _.dummy);J=jac6)
    npar = NewtonPar(tol=1e-6,max_iterations=12,verbose=true,linesearch=true)
    root = newton(prob,npar;normN=x->norm(x,Inf),callback=BK.cbMaxNorm(10.0))
    BK.converged(root) || error("BifurcationKit organizer Newton did not converge; residuals=$(root.residuals)")

    y = Float64.(root.u)
    pf = (m1=y[5],m2=y[6],m3=seed.m3); qf=y[1:4]
    final = flow_data(qf,pf;rtol=8e-12,atol=8e-14)
    closure_norm = norm(final.closure)
    event_norm = hypot(final.gp,final.gm)
    invariant_error = hypot(final.alpha-4,final.beta-4)
    mass_shift = hypot(y[5]-seed.m1,y[6]-seed.m2)
    passed = closure_norm <= 2e-8 && event_norm <= 2e-6 && invariant_error <= 5e-6 && mass_shift <= 5e-4
    passed || error("latest-stack gates failed: closure=$closure_norm events=$event_norm invariant=$invariant_error shift=$mass_shift")

    mkpath(dirname(output))
    open(output,"w") do io
        print(io,"{\n",
          "  \"implementation\": \"independent Julia 1.12.6 + BifurcationKit rank-revealed organizer Newton + Vern9 variational cross-check\",\n",
          "  \"claim_status\": \"screening_supported_independent_formulation\",\n",
          "  \"seed_name\": \"",json_string(seed.name),"\",\n",
          "  \"julia_version\": \"",VERSION,"\",\n",
          "  \"bifurcationkit_version\": \"",package_version("BifurcationKit"),"\",\n",
          "  \"selected_closure_rows\": [",join(rows,","),"],\n",
          "  \"selected_row_determinant\": ",rowdet,",\n",
          "  \"masses\": [",pf.m1,",",pf.m2,",",pf.m3,"],\n",
          "  \"chart\": [",join(qf,","),"],\n",
          "  \"closure_norm\": ",closure_norm,",\n",
          "  \"alpha\": ",final.alpha,",\n",
          "  \"beta\": ",final.beta,",\n",
          "  \"plus_one_event\": ",final.gp,",\n",
          "  \"minus_one_event\": ",final.gm,",\n",
          "  \"discriminant\": ",final.disc,",\n",
          "  \"event_norm\": ",event_norm,",\n",
          "  \"spectral_vertex_error\": ",invariant_error,",\n",
          "  \"mass_shift_from_frozen_seed\": ",mass_shift,",\n",
          "  \"newton_iterations\": ",root.itnewton,",\n",
          "  \"newton_residual_history\": [",join(root.residuals,","),"],\n",
          "  \"passed\": true\n",
          "}\n")
    end
    println("seed=",seed.name," closure=",closure_norm," event_norm=",event_norm,
            " vertex_error=",invariant_error," mass_shift=",mass_shift)
end

main()
