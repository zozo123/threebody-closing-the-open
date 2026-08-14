#!/usr/bin/env julia

# Independent publication-oriented verifier for the COM-reduced planar problem.
#
# This implementation intentionally does not import Python/NumPy/SciPy code. It
# uses Julia BigFloat arithmetic and a high-order adaptive Vern9 integrator from
# DifferentialEquations.jl.  Inputs come from the frozen Li-Li-Liao baseline.

using DifferentialEquations
using LinearAlgebra

function parse_baseline(path::String, wanted::Set{Int})
    rows = Dict{Int,NamedTuple}()
    idx = 0
    for line in eachline(path)
        f = split(strip(line))
        if length(f) != 8 || !(f[end] in ("S", "U"))
            continue
        end
        vals = try
            parse.(BigFloat, f[1:7])
        catch
            continue
        end
        idx += 1
        if idx in wanted
            rows[idx] = (
                m1=vals[1], m2=vals[2], m3=vals[3], x1=vals[4],
                v1=vals[5], v2=vals[6], period=vals[7], label=f[8],
            )
        end
    end
    rows
end

function force_and_derivative(x::AbstractVector{T}) where {T<:Real}
    r2 = x[1]^2 + x[2]^2
    r2 == 0 && error("binary collision")
    r = sqrt(r2)
    inv3 = inv(r^3)
    inv5 = inv(r^5)
    f = T[x[1]*inv3, x[2]*inv3]
    d = Matrix{T}(undef, 2, 2)
    d[1,1] = inv3 - 3*x[1]*x[1]*inv5
    d[1,2] = -3*x[1]*x[2]*inv5
    d[2,1] = -3*x[2]*x[1]*inv5
    d[2,2] = inv3 - 3*x[2]*x[2]*inv5
    return f, d
end

function addblock!(J, r::Int, c::Int, B, scale)
    @inbounds for i in 1:2, j in 1:2
        J[r+i-1, c+j-1] += scale * B[i,j]
    end
end

function reduced_rhs_jacobian(z::AbstractVector{T}, masses) where {T<:Real}
    m1, m2, m3 = masses
    q1 = @view z[1:2]
    q2 = @view z[3:4]
    d12 = T[q2[1]-q1[1], q2[2]-q1[2]]
    f1, d1 = force_and_derivative(q1)
    f2, d2 = force_and_derivative(q2)
    fd, dd = force_and_derivative(d12)

    rhs = zeros(T, 8)
    rhs[1:4] .= z[5:8]
    rhs[5:6] .= m2 .* fd .- (m1+m3) .* f1 .- m2 .* f2
    rhs[7:8] .= -m1 .* fd .- m1 .* f1 .- (m2+m3) .* f2

    J = zeros(T, 8, 8)
    J[1:4,5:8] .= Matrix{T}(I,4,4)
    addblock!(J, 5, 1, dd, -m2)
    addblock!(J, 5, 1, d1, -(m1+m3))
    addblock!(J, 5, 3, dd, m2)
    addblock!(J, 5, 3, d2, -m2)
    addblock!(J, 7, 1, dd, m1)
    addblock!(J, 7, 1, d1, -m1)
    addblock!(J, 7, 3, dd, -m1)
    addblock!(J, 7, 3, d2, -(m2+m3))
    return rhs, J
end

function augmented!(du, u, masses, t)
    z = @view u[1:8]
    rhs, J = reduced_rhs_jacobian(z, masses)
    du[1:8] .= rhs
    Phi = reshape(@view(u[9:72]), 8, 8)
    dPhi = J * Phi
    du[9:72] .= vec(dPhi)
    nothing
end

function initial_reduced(row)
    v3 = -(row.m1*row.v1 + row.m2*row.v2)/row.m3
    BigFloat[
        row.x1, 0, 1, 0,
        0, row.v1-v3, 0, row.v2-v3,
    ]
end

function verify(row; tol::BigFloat)
    z0 = initial_reduced(row)
    phi0 = Matrix{BigFloat}(I,8,8)
    u0 = vcat(z0, vec(phi0))
    masses = (row.m1,row.m2,row.m3)
    prob = ODEProblem(augmented!, u0, (BigFloat(0),row.period), masses)
    sol = solve(prob, Vern9(); reltol=tol, abstol=tol, save_everystep=false,
                maxiters=10^8)
    SciMLBase.successful_retcode(sol) || error("integration failed: $(sol.retcode)")
    uf = sol.u[end]
    zf = uf[1:8]
    M = reshape(uf[9:72],8,8)
    closure = norm(zf-z0)
    alpha = tr(M)
    beta = (alpha^2 - tr(M*M))/2
    disc = (alpha-4)^2 - 4*(beta-4*alpha+8)
    rootdisc = sqrt(Complex{BigFloat}(disc))
    t1 = ((alpha-4) + rootdisc)/2
    t2 = ((alpha-4) - rootdisc)/2
    score = min(disc, BigFloat(2)-abs(t1), BigFloat(2)-abs(t2))
    stable = disc > 0 && abs(imag(t1)) < sqrt(tol) && abs(imag(t2)) < sqrt(tol) &&
             abs(t1) < 2 && abs(t2) < 2
    return (
        closure=closure, alpha=alpha, beta=beta, disc=disc,
        t1=t1, t2=t2, score=score, stable=stable,
        steps=length(sol.t), retcode=string(sol.retcode),
    )
end

function json_complex(z)
    "[\"$(string(real(z)))\",\"$(string(imag(z)))\"]"
end

function json_result(idx, row, r)
    "{" *
    "\"baseline_row\":$(idx)," *
    "\"published_stability\":\"$(row.label)\"," *
    "\"closure_norm\":\"$(string(r.closure))\"," *
    "\"alpha\":\"$(string(r.alpha))\"," *
    "\"beta\":\"$(string(r.beta))\"," *
    "\"discriminant\":\"$(string(r.disc))\"," *
    "\"trace_roots\":[$(json_complex(r.t1)),$(json_complex(r.t2))]," *
    "\"stability_score\":\"$(string(r.score))\"," *
    "\"computed_stable\":$(r.stable)," *
    "\"agrees_with_published\":$(r.stable == (row.label == \"S\"))," *
    "\"retcode\":\"$(r.retcode)\"" *
    "}"
end

function main()
    length(ARGS) >= 2 || error("usage: verify_reduced.jl DATASET OUTPUT [DPS] [TOL_EXP] [ROWS...]")
    dataset, output = ARGS[1], ARGS[2]
    dps = length(ARGS) >= 3 ? parse(Int,ARGS[3]) : 80
    tol_exp = length(ARGS) >= 4 ? parse(Int,ARGS[4]) : 45
    row_ids = length(ARGS) >= 5 ? parse.(Int,ARGS[5:end]) : [7,12]
    bits = ceil(Int, dps*log2(10)) + 32
    setprecision(BigFloat,bits) do
        tol = parse(BigFloat,"1e-$(tol_exp)")
        rows = parse_baseline(dataset,Set(row_ids))
        length(rows) == length(row_ids) || error("missing baseline rows")
        results = String[]
        for idx in row_ids
            row = rows[idx]
            r = verify(row;tol=tol)
            println("row=",idx," published=",row.label," stable=",r.stable,
                    " closure=",r.closure," score=",r.score)
            push!(results,json_result(idx,row,r))
        end
        mkpath(dirname(output))
        open(output,"w") do io
            print(io,"{\"implementation\":\"Julia BigFloat + Vern9\",",
                  "\"dps\":",dps,",\"tolerance\":\"1e-",tol_exp,"\",",
                  "\"rows\":[",join(results,","),"]}\n")
        end
    end
end

main()
