# Search semantics and transitive invalidation

Evidence bytes and evidence meaning are separate dependencies. A matching
artifact hash proves that the bytes are the bytes previously reviewed; it does
not prove that a downstream claim is permitted by the search that produced
them.

`SEARCH_SCOPE_REGISTRY.json` is the canonical semantic contract. Every search
criterion has an explicit `/vN` identity and five boolean claim dimensions:

- whether it enumerates the historical published-label transition roots;
- whether it enumerates the full Floquet critical set;
- whether it excludes even root pairs;
- whether it excludes tangencies;
- whether its conclusion is limited to a finite sampled resolution.

New JSON search artifacts declare `search_semantics.criterion_id` and their
registry-derived claim scope. Frozen historical artifacts are not rewritten;
the registry types their exact path and sha256 instead. Completeness certificate
schema `/3` records each parent's criterion and a digest of only the criterion
definitions and release requirement on which it depends. A breaking change to
one of those definitions invalidates the certificate even if every evidence
file hash still matches, while an unrelated registry addition does not.

The graph assembler distinguishes two questions:

1. Is the bounded certificate numerically and semantically authentic?
2. Is its search scope strong enough for a full-critical-set release claim?

The current AL-pocket plus neck-raster certificate answers the first question
yes and the second no. It is valid bounded negative evidence; it does not
enumerate the full domain, exclude even roots, or exclude tangencies. Therefore
`full_critical_set_scope_passed` is false and `release_ready` cannot turn green
from the historical 620-transition substrate.

Mutation coverage changes a used criterion's description without bumping its
version. The source bytes remain untouched, and the completeness verifier must
still fail on the stale semantic-contract digest.
