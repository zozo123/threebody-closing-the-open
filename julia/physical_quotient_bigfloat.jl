#!/usr/bin/env julia

# BigFloat physical four-dimensional Floquet quotient for a canonical Jacobi
# monodromy.  This is the arbitrary-precision analogue of the structural Python
# E^omega/E construction, implemented independently in Julia.
#
# E = span{X_H, X_L} is the regular two-dimensional isotropic neutral space.
# We represent E^omega/E by the Euclidean-orthogonal complement of E inside
# E^omega.  BigFloat Householder QR constructs that four-dimensional space; no
# raw multiplier deletion is used. GenericSchur is used for eigensystems and
# singular spectra so no LAPACK Float64 demotion enters the publication path.

using LinearAlgebra
using GenericSchur

function pq_rotation_generator(z::AbstractVector{T}) where {T<:Real}
    length(z)==8 || error("canonical Jacobi state must have length 8")
    out=similar(z)
    for start in (1,3,5,7)
        x,y=z[start],z[start+1]
        out[start]=-y
        out[start+1]=x
    end
    out
end

function pq_orthonormal_neutral(flow::AbstractVector{T}, rotation::AbstractVector{T}) where {T<:Real}
    e1=copy(flow); n1=norm(e1); n1>0 || error("zero Hamiltonian flow generator")
    e1 ./= n1
    e2=rotation-dot(e1,rotation)*e1
    n2=norm(e2); n2>sqrt(eps(T)) || error("time and rotation generators are dependent")
    e2 ./= n2
    hcat(e1,e2)
end

function pq_reciprocal_pairing_error(values)
    remaining=collect(values)
    worst=zero(BigFloat)
    while !isempty(remaining)
        lam=pop!(remaining)
        isempty(remaining) && return BigFloat(Inf)
        target=inv(lam)
        distances=abs.(remaining .- target)
        idx=argmin(distances)
        mate=remaining[idx]
        deleteat!(remaining,idx)
        worst=max(worst,abs(mate-target))
    end
    worst
end

function pq_singular_values(B::AbstractMatrix{T}) where {T<:Real}
    H=transpose(B)*B
    vals=eigen(Complex{T}.(H)).values
    realvals=T[max(zero(T),real(v)) for v in vals]
    sort!(realvals,rev=true)
    sqrt.(realvals)
end

function pq_trace_invariants(A::AbstractMatrix{T}) where {T<:Real}
    a=tr(A)
    b=(a*a-tr(A*A))/2
    disc=a*a-4b+8
    plus=b-2a+2
    minus=b+2a+2
    (;a,b,disc,plus,minus)
end

function physical_quotient_bigfloat(M::AbstractMatrix{T}, z0::AbstractVector{T}, masses;
                                    rank_tolerance::T=sqrt(eps(T))) where {T<:Real}
    size(M)==(8,8) || error("canonical monodromy must be 8x8")
    length(z0)==8 || error("canonical state must have length 8")
    J=canonical_symplectic_matrix(T)
    flow,_=canonical_rhs_jacobian(z0,masses)
    rotation=pq_rotation_generator(z0)
    E=pq_orthonormal_neutral(flow,rotation)
    isotropy=opnorm(transpose(E)*J*E,Inf)

    constraints=vcat(transpose(E)*J,transpose(E)) # 4x8
    # QR of the transposed constraints gives an orthogonal basis whose first
    # four columns span the constraint row-space; the last four span W.
    F=qr(transpose(constraints))
    Q=Matrix(F.Q)
    size(Q)==(8,8) || error("expected full 8x8 QR basis, got $(size(Q))")
    R=Matrix(F.R)
    diagR=[abs(R[i,i]) for i in 1:4]
    minimum(diagR)>rank_tolerance || error("physical quotient constraints lost rank: diagR=$diagR")
    W=Q[:,5:8]
    Omega=transpose(W)*J*W
    omega_sv=pq_singular_values(Omega)
    minimum(omega_sv)>rank_tolerance || error("physical quotient symplectic form is degenerate: sv=$omega_sv")

    mapped=M*W
    span=hcat(E,W)
    leakage=opnorm(mapped-span*(transpose(span)*mapped),Inf)
    A=transpose(W)*mapped
    symplectic_defect=opnorm(transpose(A)*Omega*A-Omega,Inf)
    neutral_invariance=opnorm(M*E-E,Inf)
    eig=eigen(Complex{T}.(A))
    pairing=pq_reciprocal_pairing_error(eig.values)
    invariants=pq_trace_invariants(A)

    Bplus=A-Matrix{T}(I,4,4)
    Bminus=A+Matrix{T}(I,4,4)
    plus_sv=pq_singular_values(Bplus)
    plus2_sv=pq_singular_values(Bplus*Bplus)
    minus_sv=pq_singular_values(Bminus)
    minus2_sv=pq_singular_values(Bminus*Bminus)

    return (
        matrix=A,form=Omega,basis=W,multipliers=eig.values,eigenvectors=eig.vectors,
        isotropy_defect=isotropy,constraint_qr_diagonal=diagR,
        quotient_form_singular_values=omega_sv,leakage=leakage,
        symplectic_defect=symplectic_defect,neutral_invariance_defect=neutral_invariance,
        reciprocal_pairing_error=pairing,invariants=invariants,
        plus_singular_values=plus_sv,plus_squared_singular_values=plus2_sv,
        minus_singular_values=minus_sv,minus_squared_singular_values=minus2_sv,
    )
end
