-- Three-Body Orbit Atlas scientific catalog schema
-- Target: PostgreSQL 18 + pgvector.
--
-- The database is the authoritative query/catalog layer, NOT the sole holder of
-- arbitrary-precision numerical evidence. Publication-critical exact solver
-- payloads live in immutable content-addressed artifacts referenced by SHA-256.

BEGIN;

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TYPE verification_status AS ENUM (
    'candidate',
    'screened',
    'closure_verified',
    'variational_verified',
    'independently_reproduced',
    'release_claim'
);

CREATE TYPE stability_class AS ENUM (
    'unverified',
    'elliptic',
    'hyperbolic',
    'loxodromic',
    'marginal',
    'critical_plus_one',
    'critical_minus_one',
    'numerically_ambiguous'
);

CREATE TABLE topology_signature (
    topology_id uuid PRIMARY KEY DEFAULT uuidv7(),
    raw_free_word text,
    canonical_free_word text,
    canonical_word_sha256 char(64),
    word_length integer CHECK (word_length IS NULL OR word_length >= 0),
    syzygy_sequence text[],
    braid_type text,
    algorithm_version text NOT NULL,
    symmetry_quotient jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE (canonical_word_sha256, algorithm_version, symmetry_quotient)
);

CREATE TABLE solver_run (
    solver_run_id uuid PRIMARY KEY DEFAULT uuidv7(),
    code_revision text NOT NULL,
    implementation text NOT NULL,
    integrator text NOT NULL,
    arithmetic text NOT NULL,
    precision_digits integer NOT NULL CHECK (precision_digits > 0),
    configuration jsonb NOT NULL,
    environment jsonb NOT NULL DEFAULT '{}'::jsonb,
    started_at timestamptz NOT NULL,
    completed_at timestamptz,
    github_run_id bigint,
    artifact_manifest_sha256 char(64)
);

CREATE TABLE orbit (
    orbit_id uuid PRIMARY KEY DEFAULT uuidv7(),
    orbit_fingerprint char(64) UNIQUE,
    source text NOT NULL,
    m1_f64 double precision NOT NULL CHECK (m1_f64 > 0),
    m2_f64 double precision NOT NULL CHECK (m2_f64 > 0),
    m3_f64 double precision NOT NULL CHECK (m3_f64 > 0),
    energy_f64 double precision,
    angular_momentum_f64 double precision,
    period_f64 double precision NOT NULL CHECK (period_f64 > 0),
    initial_state_f64 double precision[] NOT NULL,
    topology_id uuid REFERENCES topology_signature(topology_id),
    exact_artifact_uri text,
    exact_artifact_sha256 char(64),
    normalization_version text NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now(),
    CHECK (cardinality(initial_state_f64) = 12),
    CHECK (
        (exact_artifact_uri IS NULL AND exact_artifact_sha256 IS NULL)
        OR (exact_artifact_uri IS NOT NULL AND exact_artifact_sha256 IS NOT NULL)
    )
);

CREATE INDEX orbit_mass_idx ON orbit (m1_f64, m2_f64, m3_f64);
CREATE INDEX orbit_period_idx ON orbit (period_f64);
CREATE INDEX orbit_topology_idx ON orbit (topology_id);

CREATE TABLE orbit_verification (
    verification_id uuid PRIMARY KEY DEFAULT uuidv7(),
    orbit_id uuid NOT NULL REFERENCES orbit(orbit_id),
    solver_run_id uuid NOT NULL REFERENCES solver_run(solver_run_id),
    status verification_status NOT NULL,
    stability stability_class NOT NULL DEFAULT 'unverified',
    closure_norm_f64 double precision NOT NULL CHECK (closure_norm_f64 >= 0),
    energy_defect_f64 double precision CHECK (energy_defect_f64 >= 0),
    angular_momentum_defect_f64 double precision CHECK (angular_momentum_defect_f64 >= 0),
    symplectic_defect_f64 double precision CHECK (symplectic_defect_f64 >= 0),
    unit_multiplier_defect_f64 double precision CHECK (unit_multiplier_defect_f64 >= 0),
    reduced_alpha_f64 double precision,
    reduced_beta_f64 double precision,
    reduced_discriminant_f64 double precision,
    reduced_trace_roots_reim_f64 double precision[],
    floquet_reim_f64 double precision[],
    monodromy_f64 double precision[],
    stability_margin_f64 double precision,
    exact_artifact_uri text,
    exact_artifact_sha256 char(64),
    notes text[] NOT NULL DEFAULT '{}',
    verified_at timestamptz NOT NULL DEFAULT now(),
    CHECK (reduced_trace_roots_reim_f64 IS NULL OR cardinality(reduced_trace_roots_reim_f64) = 4),
    CHECK (floquet_reim_f64 IS NULL OR cardinality(floquet_reim_f64) IN (16, 24)),
    CHECK (monodromy_f64 IS NULL OR cardinality(monodromy_f64) IN (64, 144)),
    CHECK (
        (exact_artifact_uri IS NULL AND exact_artifact_sha256 IS NULL)
        OR (exact_artifact_uri IS NOT NULL AND exact_artifact_sha256 IS NOT NULL)
    )
);

CREATE INDEX verification_orbit_idx ON orbit_verification (orbit_id, verified_at DESC);
CREATE INDEX verification_status_idx ON orbit_verification (status, stability);

CREATE TABLE family (
    family_id uuid PRIMARY KEY DEFAULT uuidv7(),
    canonical_name text UNIQUE,
    description text,
    continuation_definition jsonb NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE family_membership (
    family_id uuid NOT NULL REFERENCES family(family_id),
    orbit_id uuid NOT NULL REFERENCES orbit(orbit_id),
    branch_id text NOT NULL,
    pseudo_arclength_f64 double precision,
    continuation_coordinates jsonb NOT NULL,
    predictor_tangent_f64 double precision[],
    step_size_f64 double precision,
    corrector_iterations integer CHECK (corrector_iterations IS NULL OR corrector_iterations >= 0),
    parent_orbit_id uuid REFERENCES orbit(orbit_id),
    PRIMARY KEY (family_id, branch_id, orbit_id)
);

CREATE INDEX family_membership_orbit_idx ON family_membership (orbit_id);
CREATE INDEX family_membership_arc_idx ON family_membership (family_id, branch_id, pseudo_arclength_f64);

CREATE TABLE bifurcation (
    bifurcation_id uuid PRIMARY KEY DEFAULT uuidv7(),
    parent_family_id uuid NOT NULL REFERENCES family(family_id),
    child_family_id uuid REFERENCES family(family_id),
    critical_orbit_id uuid REFERENCES orbit(orbit_id),
    evidence_verification_id uuid NOT NULL REFERENCES orbit_verification(verification_id),
    bifurcation_type text NOT NULL,
    critical_parameters jsonb NOT NULL,
    uncertainty jsonb NOT NULL DEFAULT '{}'::jsonb,
    evidence_status verification_status NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now(),
    CHECK (parent_family_id <> child_family_id OR child_family_id IS NULL)
);

CREATE INDEX bifurcation_parent_idx ON bifurcation (parent_family_id);
CREATE INDEX bifurcation_child_idx ON bifurcation (child_family_id);

CREATE TABLE embedding (
    embedding_id uuid PRIMARY KEY DEFAULT uuidv7(),
    entity_type text NOT NULL CHECK (entity_type IN ('orbit', 'family', 'topology')),
    entity_id uuid NOT NULL,
    embedding_kind text NOT NULL,
    model_id text NOT NULL,
    model_version text NOT NULL,
    embedding vector,
    metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE (entity_type, entity_id, embedding_kind, model_id, model_version)
);

CREATE TABLE candidate (
    candidate_id uuid PRIMARY KEY DEFAULT uuidv7(),
    model_run_id text,
    source text NOT NULL,
    masses_f64 double precision[] NOT NULL,
    proposed_chart_f64 double precision[],
    acquisition_score double precision,
    uncertainty jsonb NOT NULL DEFAULT '{}'::jsonb,
    status verification_status NOT NULL DEFAULT 'candidate',
    promoted_orbit_id uuid REFERENCES orbit(orbit_id),
    payload jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now(),
    CHECK (cardinality(masses_f64) = 3)
);

CREATE TABLE dataset_release (
    release_id text PRIMARY KEY,
    git_revision text NOT NULL,
    schema_version text NOT NULL,
    manifest_sha256 char(64) NOT NULL UNIQUE,
    counts jsonb NOT NULL,
    known_issues jsonb NOT NULL DEFAULT '[]'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE outbox (
    event_id uuid PRIMARY KEY DEFAULT uuidv7(),
    aggregate_type text NOT NULL,
    aggregate_id uuid NOT NULL,
    event_type text NOT NULL,
    payload jsonb NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now(),
    published_at timestamptz
);

CREATE INDEX outbox_pending_idx ON outbox (created_at) WHERE published_at IS NULL;

COMMIT;
