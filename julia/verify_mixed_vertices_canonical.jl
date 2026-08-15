#!/usr/bin/env julia

# End-to-end arbitrary-precision verifier for a mixed (+1,-1) organizer.
#
# Layer 1 independently corrects the periodic orbit and solves G+=G-=0 in the
# noncanonical COM-relative formulation.  Layer 2 independently reconstructs
# canonical Jacobi coordinates, reintegrates the orbit/variational equations,
# and forms the physical E^omega/E quotient.  The final artifact therefore
# carries root localization, canonical symplectic diagnostics, physical Floquet
# events, reciprocal pairing, and Jordan-chain singular spectra in one record.

module RootLayer
include(joinpath(@__DIR__, "verify_mixed_vertices.jl"))
end

module CanonicalLayer
include(joinpath(@__DIR__, "canonical_jacobi.jl"))
include(joinpath(@__DIR__, "physical_quotient_bigfloat.jl"))
end

using LinearAlgebra

function json_complex(z)
    "[\"$(real(z))\",\"$(imag(z))\"]"
end

function json_vector(xs)
    "[" * join(("\"$(x)\"" for x in xs), ",") * "]"
end

function canonical_row(root)
    (
        m1=root.masses[1], m2=root.masses[2], m3=root.masses[3],
        x1=root.p[1], v1=root.p[2], v2=root.p[3], period=root.p[4], label="C",
    )
end

function main()
    length(ARGS)>=2 || error("usage: verify_mixed_vertices_canonical.jl SEED_TSV OUTPUT [DPS] [TOL_EXP] [CLOSURE_EXP] [EVENT_EXP]")
    seed_path,output=ARGS[1],ARGS[2]
    dps=length(ARGS)>=3 ? parse(Int,ARGS[3]) : 60
    tol_exp=length(ARGS)>=4 ? parse(Int,ARGS[4]) : 28
    closure_exp=length(ARGS)>=5 ? parse(Int,ARGS[5]) : 18
    event_exp=length(ARGS)>=6 ? parse(Int,ARGS[6]) : 12
    bits=ceil(Int,dps*log2(10))+32

    setprecision(BigFloat,bits) do
        tol=parse(BigFloat,"1e-$(tol_exp)")
        closure_target=parse(BigFloat,"1e-$(closure_exp)")
        event_target=parse(BigFloat,"1e-$(event_exp)")
        h=parse(BigFloat,"1e-6")
        max_shift=parse(BigFloat,"1e-4")
        seeds=RootLayer.parse_mixed_vertex_seeds(seed_path)
        length(seeds)==1 || error("canonical mixed verifier expects exactly one frozen seed")
        seed=only(seeds)
        p0=BigFloat[seed.x1,seed.v1,seed.v2,seed.period]
        center=RootLayer.corrected_mixed_sample(seed.m1,seed.m2,seed.m3,p0;tol=tol,target=closure_target)
        root,iters=RootLayer.solve_mixed_mass_root(
            seed,center;tol=tol,target=closure_target,event_target=event_target,h=h,
            max_shift=max_shift,maxiter=10,
        )
        events=RootLayer.mixed_events(root)
        root_event_norm=norm(events)
        root.closure<=closure_target || error("relative root closure gate failed: $(root.closure)")
        root_event_norm<=event_target || error("relative root event gate failed: $root_event_norm")

        row=canonical_row(root)
        can=CanonicalLayer.canonical_floquet(row;tol=tol)
        masses=(row.m1,row.m2,row.m3)
        z0=CanonicalLayer.full_to_canonical_jacobi(CanonicalLayer.initial_full(row),masses)
        phys=CanonicalLayer.physical_quotient_bigfloat(can.M,z0,masses)
        pinv=phys.invariants

        can.closure <= parse(BigFloat,"1e-14") || error("canonical closure gate failed: $(can.closure)")
        can.symplectic_defect <= parse(BigFloat,"1e-14") || error("canonical symplectic gate failed: $(can.symplectic_defect)")
        can.reciprocal_pairing_error <= parse(BigFloat,"1e-10") || error("canonical reciprocal-pair gate failed: $(can.reciprocal_pairing_error)")
        phys.symplectic_defect <= parse(BigFloat,"1e-10") || error("physical quotient symplectic gate failed: $(phys.symplectic_defect)")
        phys.leakage <= parse(BigFloat,"1e-10") || error("physical quotient leakage gate failed: $(phys.leakage)")
        phys.reciprocal_pairing_error <= parse(BigFloat,"1e-8") || error("physical reciprocal-pair gate failed: $(phys.reciprocal_pairing_error)")
        hypot(pinv.plus,pinv.minus) <= parse(BigFloat,"1e-8") || error("physical mixed-event gate failed: G+=$(pinv.plus) G-=$(pinv.minus)")

        mkpath(dirname(output))
        open(output,"w") do io
            print(io,
              "{\n",
              "  \"implementation\": \"independent Julia BigFloat root solve + canonical Jacobi Vern9 + physical E^omega/E quotient + GenericSchur\",\n",
              "  \"claim_status\": \"high_precision_supported; Jordan/nondegeneracy spectra included for classification\",\n",
              "  \"name\": \"",seed.name,"\",\n",
              "  \"dps\": ",dps,",\n",
              "  \"masses\": ",json_vector(root.masses),",\n",
              "  \"chart\": ",json_vector(root.p),",\n",
              "  \"relative_closure_norm\": \"",root.closure,"\",\n",
              "  \"relative_event_norm\": \"",root_event_norm,"\",\n",
              "  \"relative_plus_one_event\": \"",events[1],"\",\n",
              "  \"relative_minus_one_event\": \"",events[2],"\",\n",
              "  \"mass_newton_iterations\": ",iters,",\n",
              "  \"canonical_closure_norm\": \"",can.closure,"\",\n",
              "  \"canonical_symplectic_defect_inf\": \"",can.symplectic_defect,"\",\n",
              "  \"canonical_hamiltonian_linearization_defect_inf\": \"",can.hamiltonian_linearization_defect,"\",\n",
              "  \"canonical_reciprocal_pairing_error\": \"",can.reciprocal_pairing_error,"\",\n",
              "  \"physical_symplectic_defect_inf\": \"",phys.symplectic_defect,"\",\n",
              "  \"physical_leakage_inf\": \"",phys.leakage,"\",\n",
              "  \"physical_neutral_isotropy_defect_inf\": \"",phys.isotropy_defect,"\",\n",
              "  \"physical_neutral_invariance_defect_inf\": \"",phys.neutral_invariance_defect,"\",\n",
              "  \"physical_reciprocal_pairing_error\": \"",phys.reciprocal_pairing_error,"\",\n",
              "  \"physical_a\": \"",pinv.a,"\",\n",
              "  \"physical_b\": \"",pinv.b,"\",\n",
              "  \"physical_discriminant\": \"",pinv.disc,"\",\n",
              "  \"physical_plus_one_event\": \"",pinv.plus,"\",\n",
              "  \"physical_minus_one_event\": \"",pinv.minus,"\",\n",
              "  \"physical_multipliers\": [",join(json_complex.(phys.multipliers),","),"],\n",
              "  \"plus_singular_values\": ",json_vector(phys.plus_singular_values),",\n",
              "  \"plus_squared_singular_values\": ",json_vector(phys.plus_squared_singular_values),",\n",
              "  \"minus_singular_values\": ",json_vector(phys.minus_singular_values),",\n",
              "  \"minus_squared_singular_values\": ",json_vector(phys.minus_squared_singular_values),",\n",
              "  \"quotient_form_singular_values\": ",json_vector(phys.quotient_form_singular_values),",\n",
              "  \"passed\": true\n",
              "}\n"
            )
        end
        println("canonical mixed organizer ",seed.name," PASS event_norm=",root_event_norm,
                " physical_events=",pinv.plus,",",pinv.minus,
                " canonical_symplectic=",can.symplectic_defect,
                " physical_symplectic=",phys.symplectic_defect)
    end
end

main()
