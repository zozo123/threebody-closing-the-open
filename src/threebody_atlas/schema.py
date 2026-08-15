"""Versioned machine-readable records for candidates and verifications."""
from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field, field_validator

SCHEMA_VERSION = "atlas.orbit-verification.v1"


class VerificationStatus(StrEnum):
    CANDIDATE = "candidate"
    SCREENED = "screened"
    CLOSURE_VERIFIED = "closure_verified"
    VARIATIONAL_VERIFIED = "variational_verified"
    INDEPENDENTLY_REPRODUCED = "independently_reproduced"


class StabilityClass(StrEnum):
    UNVERIFIED = "unverified"
    ELLIPTIC = "elliptic"
    HYPERBOLIC = "hyperbolic"
    LOXODROMIC = "loxodromic"
    MARGINAL = "marginal"
    NUMERICALLY_AMBIGUOUS = "numerically_ambiguous"


class OrbitCandidate(BaseModel):
    schema_version: Literal[SCHEMA_VERSION] = SCHEMA_VERSION
    source: str
    masses: tuple[float, float, float]
    period: float = Field(gt=0)
    initial_state: tuple[float, ...]
    free_group_word: str | None = None

    @field_validator("initial_state")
    @classmethod
    def validate_state(cls, value: tuple[float, ...]) -> tuple[float, ...]:
        if len(value) != 12:
            raise ValueError("planar full-coordinate state must have 12 components")
        return value


class VerificationRecord(BaseModel):
    schema_version: Literal[SCHEMA_VERSION] = SCHEMA_VERSION
    status: VerificationStatus
    stability_class: StabilityClass = StabilityClass.UNVERIFIED
    candidate: OrbitCandidate
    closure_norm: float
    energy_defect: float
    angular_momentum_defect: float
    symplectic_defect: float | None = None
    floquet_multipliers: list[tuple[float, float]] = Field(default_factory=list)
    reduced_alpha: float | None = None
    reduced_beta: float | None = None
    reduced_discriminant: float | None = None
    reduced_trace_roots: list[tuple[float, float]] = Field(default_factory=list)
    stability_margin: float | None = None
    arithmetic: str
    precision_digits: int
    code_revision: str | None = None
    verified_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    notes: list[str] = Field(default_factory=list)
