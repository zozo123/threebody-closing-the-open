"""AI proposal engine for continuation and boundary exploration.

Predictions from this module are *candidates*.  They are never promoted to orbit
or stability evidence until the deterministic shooting and verification stack
accepts them.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

Array = np.ndarray


@dataclass(frozen=True)
class CandidateProposal:
    m1: float
    m2: float
    m3: float
    x1: float
    v1: float
    v2: float
    period: float
    continuation_uncertainty: float
    stable_probability: float
    boundary_interest: float
    acquisition_score: float


class AtlasSurrogate:
    """Extra-trees ensemble for chart warm starts and stability acquisition.

    The ensemble dispersion supplies a pragmatic epistemic-uncertainty proxy.
    It is not a calibrated posterior and is recorded as an acquisition score,
    not a physical uncertainty bar.
    """

    def __init__(self, *, n_estimators: int = 128, random_state: int = 20260814):
        try:
            from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
        except ImportError as exc:
            raise ImportError("install the optional ML dependencies with pip install -e '.[ml]'") from exc
        self._reg = ExtraTreesRegressor(
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=-1,
            min_samples_leaf=2,
        )
        self._clf = ExtraTreesClassifier(
            n_estimators=n_estimators,
            random_state=random_state + 1,
            n_jobs=-1,
            min_samples_leaf=2,
            class_weight="balanced",
        )
        self._fitted = False

    def fit(self, masses: Array, chart: Array, stable: Array) -> "AtlasSurrogate":
        masses = np.asarray(masses, dtype=float)
        chart = np.asarray(chart, dtype=float)
        stable = np.asarray(stable, dtype=int)
        if masses.ndim != 2 or masses.shape[1] != 3:
            raise ValueError("masses must have shape (n,3)")
        if chart.shape != (len(masses), 4):
            raise ValueError("chart must contain (x1,v1,v2,T) for each row")
        if stable.shape != (len(masses),):
            raise ValueError("stable labels must have shape (n,)")
        self._reg.fit(masses, chart)
        self._clf.fit(masses, stable)
        self._fitted = True
        return self

    def propose(self, masses: Array) -> list[CandidateProposal]:
        if not self._fitted:
            raise RuntimeError("fit the surrogate before proposing candidates")
        masses = np.asarray(masses, dtype=float)
        prediction = self._reg.predict(masses)
        member_predictions = np.stack([tree.predict(masses) for tree in self._reg.estimators_])
        # Normalize chart dimensions before aggregating ensemble spread so period
        # does not dominate simply because of units.
        scale = np.std(prediction, axis=0)
        scale = np.where(scale > 1e-12, scale, 1.0)
        uncertainty = np.sqrt(np.mean(np.var(member_predictions / scale, axis=0), axis=1))
        probabilities = self._clf.predict_proba(masses)
        classes = list(self._clf.classes_)
        stable_col = classes.index(1) if 1 in classes else None
        stable_probability = (
            probabilities[:, stable_col] if stable_col is not None else np.zeros(len(masses))
        )
        boundary_interest = 1.0 - 2.0 * np.abs(stable_probability - 0.5)
        # Favor points that are both poorly known and near the classifier's
        # decision boundary; the small offset still explores high-uncertainty
        # regions away from an apparent boundary.
        score = boundary_interest * (1.0 + uncertainty) + 0.1 * uncertainty
        proposals = []
        for i, mass in enumerate(masses):
            proposals.append(
                CandidateProposal(
                    m1=float(mass[0]),
                    m2=float(mass[1]),
                    m3=float(mass[2]),
                    x1=float(prediction[i, 0]),
                    v1=float(prediction[i, 1]),
                    v2=float(prediction[i, 2]),
                    period=float(prediction[i, 3]),
                    continuation_uncertainty=float(uncertainty[i]),
                    stable_probability=float(stable_probability[i]),
                    boundary_interest=float(boundary_interest[i]),
                    acquisition_score=float(score[i]),
                )
            )
        return proposals
