# ATLAS v1 god-mode execution DAG

Scientific status remains `OPEN` until Gates A/B/D pass and `#118`’s solved gate
authorizes a `solved-v*` tag. This file is the control-plane graph for finishing
the frozen problem, not a new research direction.

```mermaid
flowchart TD
    Q[Frozen v1 question] --> C[Gate C PASSED: one continuation component]
    Q --> H[Wave 0 harvest live artifacts]
    Q --> L[#117 ledger sync]

    H --> A_mix[3 mixed organizers physical-canonical PASSED]
    H --> A_run[QR-fixed +1 / collision bind still running]
    H --> D_run[Long daughter genealogy / fold BF / secondary-right E2E]

    L --> W1
    A_mix --> W1[Wave 1 unblock solvers]
    A_run --> W1
    D_run --> W1

    W1 --> I109["#109 persist + polish 620-cell localizer"]
    W1 --> I110["#110 fold descent: 0.997 to 0.996 PA fallback"]
    W1 --> I113["#113 BigFloat daughter on d0-minus"]
    W1 --> I112["#112 freeze mixed Sp4 records; finish exact +1 / HH bind"]

    I109 --> W2[Wave 2 graph completion]
    I110 --> W2
    I112 --> W2
    I113 --> W2

    W2 --> I114["#114 attach +1/-1 arcs through mixed nodes"]
    W2 --> I111["#111 classify secondary-right death"]
    W2 --> I115["#115 neck rerun + AL/vertex harvest"]

    I114 --> I116["#116 assemble machine-readable critical graph"]
    I111 --> I116
    I115 --> I116
    I113 --> I116

    I116 --> I118["#118 novelty + manuscript + require-solved"]
    I118 -->|all gates pass| SOLVED[solved-v* tag]
    I118 -->|any blocker remains| OPEN[remain OPEN]
```

## Wave rules

1. Harvest before launching. Do not duplicate secondary-right / fold-BigFloat / long-genealogy jobs that are already running.
2. Never loosen a scientific gate to get a green check.
3. `Newton failed` is not an endpoint class.
4. Do not mark `SOLVED` from a denser plot, an ML proposal, or a float64-only collision.

## What this branch attacks immediately

- `#117` / Gate C bookkeeping: ledger and discovery manifest now match the passed connectivity certificate.
- `#112` harvest: the three mixed-canonical Actions artifacts are frozen under `research/evidence/` with `passed: true`.
- `#109`: persist every 620-cell attempt; polish the event with a 1-D `m2` Newton; keep the `2e-8` event gate.
- `#110`: if fixed-`m1` descent dies after one point, start hybrid continuation from the published `0.997 → 0.996` pair and still write a JSON.
- `#113`: independently correct the physical-soft **minus** representative, not the near-switch plus seeds that already stalled at `1e-9`.
