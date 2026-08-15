#!/usr/bin/env julia

# Bind the independently corrected BigFloat critical brackets directly to the
# canonical Hamiltonian / physical Floquet mechanism classification.
#
# The relative-coordinate layer only locates a periodic-orbit event bracket.
# Each flank is then reconstructed independently in canonical Jacobi variables,
# reintegrated with Vern9, reduced to the physical E^omega/E quotient, and
# classified from the physical Sp(4) trace roots and Krein forms.

module RelativeLayer
include(joinpath(@__DIR__, "verify_critical_points.jl"))
end

module CanonicalLayer
include(joinpath(@__DIR__, "canonical_jacobi.jl"))
include(joinpath(@__DIR__, "physical_quotient_bigfloat.jl"))
end

using LinearAlgebra

function canonical_row(sample)
    (
        m1=sample.masses[1], m2=sample.masses[2], m3=sample.masses[3],
        x1=sample.p[1], v1=sample.p[2], v2=sample.p[3], period=sample.p[4], label="C",
    )
end

function physical_trace_roots(inv)
    disc=Complex{BigFloat}(inv.disc)
    root=sqrt(disc)
    ((Complex{BigFloat}(inv.a)+root)/2,
     (Complex{BigFloat}(inv.a)-root)/2)
end

function physical_krein_upper_half(phys; unit_tolerance::BigFloat)
    J=Complex{BigFloat}.(phys.form)
    out=Tuple{Complex{BigFloat},BigFloat}[]
    for j in eachindex(phys.multipliers)
        lam=phys.multipliers[j]
        if imag(lam)>0 && abs(abs(lam)-1)<=unit_tolerance
            v=phys.eigenvectors[:,j]
            form=real(-Complex{BigFloat}(0,1)*dot(v,J*v))
            push!(out,(lam,form))
        end
    end
    out
end

function side_diagnostics(sample;tol::BigFloat)
    row=canonical_row(sample)
    can=CanonicalLayer.canonical_floquet(row;tol=tol)
    masses=(row.m1,row.m2,row.m3)
    z0=CanonicalLayer.full_to_canonical_jacobi(CanonicalLayer.initial_full(row),masses)
    phys=CanonicalLayer.physical_quotient_bigfloat(can.M,z0,masses)
    roots=physical_trace_roots(phys.invariants)
    unit_tol=max(sqrt(tol),parse(BigFloat,"1e-18"))
    krein=physical_krein_upper_half(phys;unit_tolerance=unit_tol)
    radii=abs.(phys.multipliers)
    (
        sample=sample,can=can,phys=phys,roots=roots,krein=krein,
        max_radius_deviation=maximum(abs.(radii.-1)),
    )
end

function classify_plus_one(a,b)
    # A generic physical +1 boundary has one trace root crossing t=+2 while
    # the other remains away from +2.  The two corrected flanks must have
    # opposite signs of the exact Sp(4) P(+2) event.
    ga=a.phys.invariants.plus
    gb=b.phys.invariants.plus
    signchange=ga==0 || gb==0 || sign(ga)!=sign(gb)
    distances_a=sort(abs.(BigFloat[real(a.roots[1])-2,real(a.roots[2])-2]))
    distances_b=sort(abs.(BigFloat[real(b.roots[1])-2,real(b.roots[2])-2]))
    other_separated=min(distances_a[2],distances_b[2])>parse(BigFloat,"1e-3")
    signchange && other_separated || error(
        "physical +1 mechanism gate failed: G+=($ga,$gb), other-root distances=$(distances_a[2]),$(distances_b[2])"
    )
    "generic_physical_plus_one_crossing"
end

function classify_trace_collision(a,b)
    da=a.phys.invariants.disc
    db=b.phys.invariants.disc
    signchange=da==0 || db==0 || sign(da)!=sign(db)
    signchange || error("physical discriminant does not change sign: $da, $db")

    stable = da>0 ? a : b
    unstable = da<0 ? a : b
    stable.phys.invariants.disc>0 || error("no positive-discriminant flank")
    unstable.phys.invariants.disc<0 || error("no negative-discriminant flank")

    # Positive discriminant flank: two distinct unit-circle physical pairs.
    length(stable.krein)==2 || error("expected two upper-half unit-circle physical modes, got $(length(stable.krein))")
    k1,k2=stable.krein[1][2],stable.krein[2][2]
    min(abs(k1),abs(k2))>parse(BigFloat,"1e-12") || error("Krein form too small for sign classification: $k1,$k2")
    k1*k2<0 || error("stable collision flank does not have opposite Krein signs: $k1,$k2")

    # Negative physical trace discriminant is the Sp(4) signature of a
    # reciprocal complex quartet.  Also require the numerically computed
    # multipliers to have moved measurably off the unit circle.
    unstable.max_radius_deviation>parse(BigFloat,"1e-10") || error(
        "negative-discriminant flank did not open into a measurable complex quartet: $(unstable.max_radius_deviation)"
    )
    "opposite_krein_hamiltonian_hopf_collision"
end

function json_complex(z)
    "[\"$(real(z))\",\"$(imag(z))\"]"
end

function json_krein(k)
    "[" * join(("{\"multiplier\":"*json_complex(x[1])*",\"form\":\"$(x[2])\"}" for x in k),",") * "]"
end

function side_json(d,mode)
    inv=d.phys.invariants
    "{" *
    "\"masses\":[\"$(d.sample.masses[1])\",\"$(d.sample.masses[2])\",\"$(d.sample.masses[3])\"]," *
    "\"relative_closure_norm\":\"$(d.sample.closure)\"," *
    "\"relative_event\":\"$(RelativeLayer.critical_event(d.sample,mode))\"," *
    "\"canonical_closure_norm\":\"$(d.can.closure)\"," *
    "\"canonical_symplectic_defect_inf\":\"$(d.can.symplectic_defect)\"," *
    "\"canonical_reciprocal_pairing_error\":\"$(d.can.reciprocal_pairing_error)\"," *
    "\"physical_symplectic_defect_inf\":\"$(d.phys.symplectic_defect)\"," *
    "\"physical_leakage_inf\":\"$(d.phys.leakage)\"," *
    "\"physical_reciprocal_pairing_error\":\"$(d.phys.reciprocal_pairing_error)\"," *
    "\"physical_a\":\"$(inv.a)\",\"physical_b\":\"$(inv.b)\"," *
    "\"physical_discriminant\":\"$(inv.disc)\"," *
    "\"physical_plus_one_event\":\"$(inv.plus)\",\"physical_minus_one_event\":\"$(inv.minus)\"," *
    "\"physical_trace_roots\":["*json_complex(d.roots[1])*","*json_complex(d.roots[2])*"]," *
    "\"physical_multipliers\":["*join(json_complex.(d.phys.multipliers),",")*"]," *
    "\"upper_half_unit_krein\":"*json_krein(d.krein)*"," *
    "\"max_multiplier_radius_deviation\":\"$(d.max_radius_deviation)\"" *
    "}"
end

function main()
    length(ARGS)>=2 || error("usage: verify_critical_events_canonical.jl SEED_TSV OUTPUT [DPS] [TOL_EXP] [CLOSURE_EXP] [WIDTH_EXP]")
    seed_path,output=ARGS[1],ARGS[2]
    dps=length(ARGS)>=3 ? parse(Int,ARGS[3]) : 60
    tol_exp=length(ARGS)>=4 ? parse(Int,ARGS[4]) : 28
    closure_exp=length(ARGS)>=5 ? parse(Int,ARGS[5]) : 18
    width_exp=length(ARGS)>=6 ? parse(Int,ARGS[6]) : 8
    bits=ceil(Int,dps*log2(10))+32

    setprecision(BigFloat,bits) do
        tol=parse(BigFloat,"1e-$(tol_exp)")
        target=parse(BigFloat,"1e-$(closure_exp)")
        width=parse(BigFloat,"1e-$(width_exp)")
        seeds=RelativeLayer.parse_critical_seeds(seed_path)
        length(seeds)==1 || error("canonical critical verifier expects exactly one frozen seed")
        seed=only(seeds)
        p0=BigFloat[seed.x1,seed.v1,seed.v2,seed.period]
        center=RelativeLayer.corrected_sample(seed.m1,seed.m2,seed.m3,p0;tol=tol,target=target)
        left,right,_,_=RelativeLayer.locate_local_bracket(
            seed,center;tol=tol,target=target,initial_halfwidth=parse(BigFloat,"2e-5"),max_expansions=5,
        )
        lo,hi,iters=RelativeLayer.refine_event_edge(left,right,seed.event_mode;tol=tol,target=target,width=width)
        max(lo.closure,hi.closure)<=target || error("relative closure gate failed")
        hi.masses[2]-lo.masses[2]<=width || error("relative bracket width gate failed")

        # Mechanism flanks must be far enough from the root to resolve the
        # eigenvalue topology but remain in the same local event neighborhood.
        root_mid=(lo.masses[2]+hi.masses[2])/2
        flank_offset=parse(BigFloat,"5e-6")
        pmean=(lo.p+hi.p)/2
        flank_lo=RelativeLayer.corrected_sample(seed.m1,root_mid-flank_offset,seed.m3,pmean;tol=tol,target=target)
        flank_hi=RelativeLayer.corrected_sample(seed.m1,root_mid+flank_offset,seed.m3,pmean;tol=tol,target=target)
        a=side_diagnostics(flank_lo;tol=tol)
        b=side_diagnostics(flank_hi;tol=tol)

        for d in (a,b)
            d.can.closure<=parse(BigFloat,"1e-14") || error("canonical closure gate failed: $(d.can.closure)")
            d.can.symplectic_defect<=parse(BigFloat,"1e-14") || error("canonical symplectic gate failed: $(d.can.symplectic_defect)")
            d.can.reciprocal_pairing_error<=parse(BigFloat,"1e-10") || error("canonical reciprocal gate failed: $(d.can.reciprocal_pairing_error)")
            d.phys.symplectic_defect<=parse(BigFloat,"1e-10") || error("physical symplectic gate failed: $(d.phys.symplectic_defect)")
            d.phys.leakage<=parse(BigFloat,"1e-10") || error("physical leakage gate failed: $(d.phys.leakage)")
            d.phys.reciprocal_pairing_error<=parse(BigFloat,"1e-8") || error("physical reciprocal gate failed: $(d.phys.reciprocal_pairing_error)")
        end

        mechanism = seed.event_mode=="plus_one" ? classify_plus_one(a,b) :
                    seed.event_mode=="trace_collision" ? classify_trace_collision(a,b) :
                    error("unsupported release mechanism in this verifier: $(seed.event_mode)")

        mkpath(dirname(output))
        open(output,"w") do io
            print(io,
              "{\n",
              "  \"implementation\": \"independent Julia BigFloat event bracket + canonical Jacobi Vern9 + physical E^omega/E quotient + GenericSchur\",\n",
              "  \"claim_status\": \"independently_reproduced canonical physical critical mechanism\",\n",
              "  \"name\": \"",seed.name,"\",\n",
              "  \"event_mode\": \"",seed.event_mode,"\",\n",
              "  \"mechanism\": \"",mechanism,"\",\n",
              "  \"dps\": ",dps,",\n",
              "  \"critical_bracket_m2\": [\"",lo.masses[2],"\",\"",hi.masses[2],"\"],\n",
              "  \"critical_bracket_width\": \"",hi.masses[2]-lo.masses[2],"\",\n",
              "  \"critical_bracket_events\": [\"",RelativeLayer.critical_event(lo,seed.event_mode),"\",\"",RelativeLayer.critical_event(hi,seed.event_mode),"\"],\n",
              "  \"refinement_iterations\": ",iters,",\n",
              "  \"flank_offset_m2\": \"",flank_offset,"\",\n",
              "  \"lower_flank\": ",side_json(a,seed.event_mode),",\n",
              "  \"upper_flank\": ",side_json(b,seed.event_mode),",\n",
              "  \"passed\": true\n",
              "}\n"
            )
        end
        println("canonical critical event ",seed.name," PASS mechanism=",mechanism,
                " bracket=[",lo.masses[2],",",hi.masses[2],"]")
    end
end

main()
