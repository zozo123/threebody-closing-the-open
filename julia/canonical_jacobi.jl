#!/usr/bin/env julia

# Independent canonical translation-reduced Hamiltonian verifier.
#
# This implementation deliberately duplicates neither the Python canonical code
# nor the noncanonical relative-coordinate equations.  It uses Jacobi canonical
# coordinates z=(rho,lambda,p_rho,p_lambda), BigFloat arithmetic, Vern9, and
# GenericSchur for arbitrary-precision eigensystems.  Its purpose is to audit
# symplectic structure and Krein signatures before assigning Hamiltonian event
# labels to critical Floquet collisions.

using SciMLBase
using OrdinaryDiffEqVerner
using LinearAlgebra
using GenericSchur

function canonical_symplectic_matrix(::Type{T}=BigFloat) where {T<:Real}
    J = zeros(T,8,8)
    J[1:4,5:8] .= Matrix{T}(I,4,4)
    J[5:8,1:4] .= -Matrix{T}(I,4,4)
    J
end

function g_and_dg(x::AbstractVector{T}) where {T<:Real}
    r2 = x[1]^2 + x[2]^2
    r2 == 0 && error("binary collision")
    r = sqrt(r2)
    inv3 = inv(r^3)
    inv5 = inv(r^5)
    g = T[x[1]*inv3,x[2]*inv3]
    D = Matrix{T}(undef,2,2)
    D[1,1] = inv3 - 3*x[1]^2*inv5
    D[1,2] = -3*x[1]*x[2]*inv5
    D[2,1] = D[1,2]
    D[2,2] = inv3 - 3*x[2]^2*inv5
    g,D
end

function initial_full(row)
    v3 = -(row.m1*row.v1 + row.m2*row.v2)/row.m3
    BigFloat[
        row.x1,0,
        1,0,
        0,0,
        0,row.v1,
        0,row.v2,
        0,v3,
    ]
end

function full_to_canonical_jacobi(full::AbstractVector{T}, masses) where {T<:Real}
    m1,m2,m3 = masses
    m12 = m1+m2
    mt = m12+m3
    mu12 = m1*m2/m12
    mu3 = m3*m12/mt
    r1,r2,r3 = @view(full[1:2]),@view(full[3:4]),@view(full[5:6])
    v1,v2,v3 = @view(full[7:8]),@view(full[9:10]),@view(full[11:12])
    rho = T[r2[1]-r1[1],r2[2]-r1[2]]
    c12 = T[(m1*r1[1]+m2*r2[1])/m12,(m1*r1[2]+m2*r2[2])/m12]
    v12 = T[(m1*v1[1]+m2*v2[1])/m12,(m1*v1[2]+m2*v2[2])/m12]
    lam = T[r3[1]-c12[1],r3[2]-c12[2]]
    prho = mu12 .* T[v2[1]-v1[1],v2[2]-v1[2]]
    plam = mu3 .* T[v3[1]-v12[1],v3[2]-v12[2]]
    vcat(rho,lam,prho,plam)
end

function canonical_rhs_jacobian(z::AbstractVector{T}, masses) where {T<:Real}
    m1,m2,m3 = masses
    m12 = m1+m2
    mt = m12+m3
    mu12 = m1*m2/m12
    mu3 = m3*m12/mt
    a = m2/m12
    b = m1/m12

    rho = @view z[1:2]
    lam = @view z[3:4]
    prho = @view z[5:6]
    plam = @view z[7:8]
    x13 = T[lam[1]+a*rho[1],lam[2]+a*rho[2]]
    x23 = T[lam[1]-b*rho[1],lam[2]-b*rho[2]]
    g12,d12 = g_and_dg(rho)
    g13,d13 = g_and_dg(x13)
    g23,d23 = g_and_dg(x23)

    grad_rho = m1*m2 .* g12 .+ m1*m3*a .* g13 .- m2*m3*b .* g23
    grad_lam = m1*m3 .* g13 .+ m2*m3 .* g23

    rhs = zeros(T,8)
    rhs[1:2] .= prho ./ mu12
    rhs[3:4] .= plam ./ mu3
    rhs[5:6] .= -grad_rho
    rhs[7:8] .= -grad_lam

    hrr = m1*m2 .* d12 .+ m1*m3*a*a .* d13 .+ m2*m3*b*b .* d23
    hll = m1*m3 .* d13 .+ m2*m3 .* d23
    hrl = m1*m3*a .* d13 .- m2*m3*b .* d23

    A = zeros(T,8,8)
    A[1:2,5:6] .= Matrix{T}(I,2,2) ./ mu12
    A[3:4,7:8] .= Matrix{T}(I,2,2) ./ mu3
    A[5:6,1:2] .= -hrr
    A[5:6,3:4] .= -hrl
    A[7:8,1:2] .= -transpose(hrl)
    A[7:8,3:4] .= -hll
    rhs,A
end

function canonical_augmented!(du,u,masses,t)
    z = @view u[1:8]
    rhs,A = canonical_rhs_jacobian(z,masses)
    du[1:8] .= rhs
    Phi = reshape(@view(u[9:72]),8,8)
    du[9:72] .= vec(A*Phi)
    nothing
end

function integrate_canonical(z0,masses,period;tol::BigFloat)
    phi0 = Matrix{BigFloat}(I,8,8)
    u0 = vcat(z0,vec(phi0))
    prob = ODEProblem(canonical_augmented!,u0,(BigFloat(0),period),masses)
    sol = solve(prob,Vern9();reltol=tol,abstol=tol,save_everystep=false,maxiters=10^8)
    SciMLBase.successful_retcode(sol) || error("canonical integration failed: $(sol.retcode)")
    uf=sol.u[end]
    uf[1:8],reshape(uf[9:72],8,8),sol
end

function reciprocal_pairing_error(values)
    remaining = collect(values)
    worst = zero(BigFloat)
    while !isempty(remaining)
        lam = pop!(remaining)
        isempty(remaining) && return BigFloat(Inf)
        target = inv(lam)
        distances = abs.(remaining .- target)
        idx = argmin(distances)
        mate = remaining[idx]
        deleteat!(remaining,idx)
        worst = max(worst,abs(mate-target))
    end
    worst
end

function krein_form(v,J)
    real(-Complex{BigFloat}(0,1) * dot(v,Complex{BigFloat}.(J)*v))
end

function canonical_floquet(row;tol::BigFloat)
    masses=(row.m1,row.m2,row.m3)
    z0=full_to_canonical_jacobi(initial_full(row),masses)
    zf,M,sol=integrate_canonical(z0,masses,row.period;tol=tol)
    closure=norm(zf-z0)
    J=canonical_symplectic_matrix(BigFloat)
    defect=opnorm(transpose(M)*J*M-J,Inf)
    hamiltonian_defect=begin
        _,A=canonical_rhs_jacobian(z0,masses)
        opnorm(transpose(A)*J+J*A,Inf)
    end

    # GenericSchur extends LinearAlgebra.eigen for Complex{BigFloat} matrices.
    E=eigen(Complex{BigFloat}.(M))
    values=E.values
    vectors=E.vectors
    pairing=reciprocal_pairing_error(values)
    krein=BigFloat[]
    for j in eachindex(values)
        lam=values[j]
        if abs(abs(lam)-1) <= sqrt(tol) && abs(lam-1) > sqrt(tol)
            push!(krein,krein_form(vectors[:,j],J))
        else
            push!(krein,BigFloat(NaN))
        end
    end
    (
        closure=closure,M=M,values=values,vectors=vectors,
        symplectic_defect=defect,hamiltonian_linearization_defect=hamiltonian_defect,
        reciprocal_pairing_error=pairing,krein=krein,
        retcode=string(sol.retcode),
    )
end

function canonical_json_complex(z)
    "[\"$(real(z))\",\"$(imag(z))\"]"
end

function canonical_json_real(x)
    isnan(x) ? "null" : "\"$(x)\""
end

function canonical_json_result(idx,row,r)
    vals="["*join(canonical_json_complex.(r.values),",")*"]"
    krein="["*join(canonical_json_real.(r.krein),",")*"]"
    "{" *
    "\"baseline_row\":$(idx),\"published_stability\":\"$(row.label)\"," *
    "\"closure_norm\":\"$(r.closure)\"," *
    "\"symplectic_defect_inf\":\"$(r.symplectic_defect)\"," *
    "\"hamiltonian_linearization_defect_inf\":\"$(r.hamiltonian_linearization_defect)\"," *
    "\"reciprocal_pairing_error\":\"$(r.reciprocal_pairing_error)\"," *
    "\"multipliers\":$(vals),\"unit_circle_krein_forms\":$(krein)," *
    "\"retcode\":\"$(r.retcode)\"}"
end
