#!/usr/bin/env julia

include(joinpath(@__DIR__,"verify_reduced.jl"))
include(joinpath(@__DIR__,"canonical_jacobi.jl"))

function main_canonical()
    length(ARGS) >= 2 || error("usage: verify_canonical_rows.jl DATASET OUTPUT [DPS] [TOL_EXP] [ROWS...]")
    dataset,output=ARGS[1],ARGS[2]
    dps=length(ARGS)>=3 ? parse(Int,ARGS[3]) : 70
    tol_exp=length(ARGS)>=4 ? parse(Int,ARGS[4]) : 35
    row_ids=length(ARGS)>=5 ? parse.(Int,ARGS[5:end]) : [7,12]
    bits=ceil(Int,dps*log2(10))+32
    setprecision(BigFloat,bits) do
        tol=parse(BigFloat,"1e-$(tol_exp)")
        rows=parse_baseline(dataset,Set(row_ids))
        length(rows)==length(row_ids) || error("missing baseline rows")
        results=String[]
        for idx in row_ids
            row=rows[idx]
            r=canonical_floquet(row;tol=tol)
            println("canonical row=",idx,
                    " closure=",r.closure,
                    " symplectic_defect=",r.symplectic_defect,
                    " pairing_error=",r.reciprocal_pairing_error)
            push!(results,canonical_json_result(idx,row,r))
        end
        mkpath(dirname(output))
        open(output,"w") do io
            print(io,"{\"implementation\":\"Julia BigFloat canonical Jacobi + Vern9 + GenericSchur\",",
                  "\"dps\":",dps,",\"tolerance\":\"1e-",tol_exp,"\",",
                  "\"rows\":[",join(results,","),"]}\n")
        end
    end
end

main_canonical()
