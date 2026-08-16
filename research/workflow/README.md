# Fail-closed scientific workflow integrity

The workflow-integrity core treats distributed numerical execution as an evidence
system, not as a collection of green job badges.

## Frozen rules

1. A campaign identity hashes the exact source commit, scientific specification,
   gate manifest, environment lock, task parameters, arithmetic, precision, platform,
   implementation, and every input artifact digest.
2. A campaign plan freezes sorted logical task IDs and affected claims before work
   starts.
3. The reducer requires exactly one successful result for every planned task.
   Missing, failed, stale, unexpected, duplicate, or conflicting results emit
   structured incidents and make the ledger ineligible for release.
4. Completion order cannot change a ledger or its payload-set identity.
5. A cache entry is accepted only when its full embedded scientific identity equals
   the expected identity and its key hashes that identity.
6. Evidence promotion uses immutable content-addressed artifacts/records plus one
   fsynced atomically replaced pointer. Promotion is one stage at a time and uses a
   compare-and-swap token under an exclusive file lock.
7. A crash before pointer replacement leaves the old state valid. A crash after
   replacement leaves the new state complete and valid. Partial evidence is never
   current.

The promotion sequence is `candidate -> screening -> correction -> independent ->
validated`. A complete campaign ledger is required at every transition.

## Frozen artifacts

- `V1_WORKFLOW_FAULT_MODEL.json` defines each injected fault and its only allowed
  fail-closed response.
- `V1_SCIENTIFIC_CHAOS_MATRIX_2026-08-16.json` executes worker, upload, task-set,
  parser, retry, cancellation, and ordering faults.
- `V1_FAIL_CLOSED_RELEASE_AUDIT_2026-08-16.json` executes incomplete-ledger,
  promotion-stage, crash-boundary, race, and corruption faults.
- `V1_CACHE_POISONING_AUDIT_2026-08-16.json` mutates every scientifically relevant
  cache-identity dimension.

## Scope boundary

This is the reusable safety core for issue #172. Existing campaign-specific reducers
must still migrate their shard plans/results onto these identities and ledgers. Until
that happens, the artifacts do not claim that every historical workflow is exactly
accounted or transactionally promoted.
