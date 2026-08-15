#!/usr/bin/env julia

include(joinpath(@__DIR__,"verify_reduced.jl"))
include(joinpath(@__DIR__,"canonical_jacobi.jl"))

function nontrivial_indices(r; symmetry_radius::BigFloat=BigFloat("1e-3"))
    [i for i in eachindex(r.values) if abs(r.values[i]-1) > symmetry_radius]
end

function upper_half_krein(r; symmetry_radius::BigFloat=BigFloat("1e-3"))
    out=Tuple{Complex{BigFloat},BigFloat}[]
    for i in nontrivial_indices(r;symmetry_radius=symmetry_radius)
        lam=r.values[i]
        if imag(lam) > 0 && abs(abs(lam)-1) < BigFloat("1e-8") && !isnan(r.krein[i])
            push!(out,(lam,r.krein[i]))
        end
    end
    out
end

function numerical_acceptance(idx,row,r)
    max_closure=BigFloat("1e-8")
    max_symplectic=BigFloat("1e-15")
    max_pairing=BigFloat("1e-15")
    max_hamiltonian=BigFloat("1e-30")
    r.closure <= max_closure || error("row $idx canonical closure gate failed: $(r.closure)")
    r.symplectic_defect <= max_symplectic || error("row $idx symplectic gate failed: $(r.symplectic_defect)")
    r.reciprocal_pairing_error <= max_pairing || error("row $idx reciprocal-pairing gate failed: $(r.reciprocal_pairing_error)")
    r.hamiltonian_linearization_defect <= max_hamiltonian || error(
        "row $idx Hamiltonian linearization gate failed: $(r.hamiltonian_linearization_defect)"
    )

    ids=nontrivial_indices(r)
    length(ids)==4 || error("row $idx expected four nontrivial multipliers, got $(length(ids))")
    radial=[abs(abs(r.values[i])-1) for i in ids]
    if row.label == "S"
        maximum(radial) <= BigFloat("1e-8") || error(
            "row $idx published stable but nontrivial multipliers leave unit circle: $(maximum(radial))"
        )
    else
        maximum(radial) >= BigFloat("1e-4") || error(
            "row $idx published unstable but no resolved nontrivial radial departure: $(maximum(radial))"
        )
    end
    true
end

function main_canonical()
    length(ARGS) >= 2 || error("usage: verify_canonical_rows.jl DATASET OUTPUT [DPS] [TOL_EXP] [ROWS...]")
    dataset,output=ARGS[1],ARGS[2]
    dps=length(ARGS)>=3 ? parse(Int,ARGS[3]) : 70
    tol_exp=length(ARGS)>=4 ? parse(Int,ARGS[4]) : 35
    row_ids=length(ARGS)>=5 ? parse.(Int,ARGS[5:end]) : [7,11,12]
    bits=ceil(Int,dps*log2(10))+32
    setprecision(BigFloat,bits) do
        tol=parse(BigFloat,"1e-$(tol_exp)")
        rows=parse_baseline(dataset,Set(row_ids))
        length(rows)==length(row_ids) || error("missing baseline rows")
        results=String[]
        for idx in row_ids
            row=rows[idx]
            r=canonical_floquet(row;tol=tol)
            numerical_acceptance(idx,row,r)
            krein=upper_half_krein(r)
            println("canonical row=",idx,
                    " published=",row.label,
                    " closure=",r.closure,
                    " symplectic_defect=",r.symplectic_defect,
                    " pairing_error=",r.reciprocal_pairing_error,
                    " upper_half_krein=",krein)
            push!(results,canonical_json_result(idx,row,r))
        end
        mkpath(dirname(output))
        open(output,"w") do io
            print(io,"{\"implementation\":\"Julia BigFloat canonical Jacobi + Vern9 + GenericSchur\",",
                  "\"dps\":",dps,",\"tolerance\":\"1e-",tol_exp,"\",",
                  "\"numerical_acceptance_gates\":{",
                  "\"closure\":\"1e-8\",\"symplectic_defect\":\"1e-15\",",
                  "\"reciprocal_pairing_error\":\"1e-15\",\"hamiltonian_linearization_defect\":\"1e-30\"},",
                  "\"rows\":[",join(results,","),"]}\n")
        end
    end
end

main_canonical()
