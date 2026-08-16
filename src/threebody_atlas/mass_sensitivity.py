"""Integrated variational sensitivities of the Floquet events with respect to mass.

Motivation
----------
``julia/verify_secondary_minus_fold.jl`` obtains ``dG-/dm2`` and ``dG-/dm1`` from
five-point stencils whose *nodes are complete independent periodic corrections*.
At dps=60 one corrected node is ~5.5 min, one stencil ~27 min, and ``audit_fold``
(four step sizes x two stencils) ~154 min -- which is why three CI runs died on a
120 min job wall.  Every one of those nodes re-derives, by differencing, a
quantity that is already available as a by-product of the integration we perform
anyway.  This module computes it directly.

Nothing here loosens a gate and nothing here writes evidence.  It is a
computational replacement for a finite difference, validated against that finite
difference.

Coordinates
-----------
Reduced (COM-quotient) state, identical to ``reduced.py`` and to
``julia/verify_reduced.jl``::

    z = (q1, q2, u1, u2),  q_i = r_i - r_3,  u_i = v_i - v_3   (each in R^2)

With the Newtonian kernel ``g(x) = x / |x|^3`` and ``d = q2 - q1``::

    q1' = u1
    q2' = u2
    u1' = m2 g(d) - (m1 + m3) g(q1) - m2 g(q2)                        (1)
    u2' = -m1 g(d) - m1 g(q1) - (m2 + m3) g(q2)

This is the vector field ``f(z, m)``.  Its state Jacobian ``A = D_z f`` has the
block form (``Dg`` is the 2x2 Jacobian of ``g``)::

    A[u1, q1] = -m2 Dg(d) - (m1 + m3) Dg(q1)
    A[u1, q2] =  m2 Dg(d) - m2 Dg(q2)
    A[u2, q1] =  m1 Dg(d) - m1 Dg(q1)                                 (2)
    A[u2, q2] = -m1 Dg(d) - (m2 + m3) Dg(q2)
    A[q,  u ] = I_4

Does the COM reduction contribute an implicit mass dependence?
--------------------------------------------------------------
``reduced.py`` builds the reduced field as ``R f_12(P(m) z, m)`` where ``P(m)``
is the COM-zero reconstruction (mass dependent!) and ``R`` takes differences.
The mass dependence of ``P`` cancels *identically*: ``R`` only ever reads
position/velocity differences, and the 12D accelerations depend on positions only
through the differences ``r_i - r_j``, which ``P(m) z`` reproduces independently
of ``m``.  Formally

    d/dm_i [ R f_12(P(m) z, m) ] = R [ D_x f_12 . (dP/dm_i) z ] + R (df_12/dm_i)
                                 = 0 + df/dm_i as read off (1),

because ``(dP/dm_i) z`` is a *pure translation* of all three bodies (and of all
three velocities) and ``D_x f_12`` annihilates translations.  So there is no
hidden reduction term in ``df/dm``; ``test_mass_sensitivity.py`` checks this
numerically against ``reduced.reduced_rhs``.  Reading ``df/dm`` off (1) directly:

    df/dm1 = (0, 0, -g(q1), -g(d) - g(q1))
    df/dm2 = (0, 0,  g(d) - g(q2), -g(q2))                            (3)

(m3 is held fixed, exactly as the Julia fold solver holds it fixed.)

Where the masses *do* enter through a gauge choice is the shooting chart.  The
Li--Li--Liao chart is

    p = (x1, v1, v2, T),
    z0 = c(p, m) = (x1, 0, 1, 0, 0, v1 - v3, 0, v2 - v3),
    v3 = -(m1 v1 + m2 v2) / m3        <- zero total momentum, mass dependent

so ``dc/dm`` is NOT zero:

    dc/dm1 = (0,0,0,0, 0, v1/m3, 0, v1/m3)
    dc/dm2 = (0,0,0,0, 0, v2/m3, 0, v2/m3)                            (4)

Dropping (4) is the single easiest way to get a plausible-looking but wrong
mass derivative; ``test_mass_sensitivity.py`` pins it.

Sensitivity equations
---------------------
Let ``z(t)`` solve ``z' = f(z, m)``, ``Phi(t) = dz(t)/dz0`` the usual
fundamental matrix, and define

    S_i(t) = dz(t)/dm_i        (8-vector)
    Psi_i(t) = dPhi(t)/dm_i    (8x8)

Differentiating ``z' = f`` and ``Phi' = A Phi`` with respect to ``m_i``:

    S_i'   = A S_i + df/dm_i,                          S_i(0) = dz0/dm_i     (5)
    Psi_i' = A Psi_i + [ (D_z A . S_i) + dA/dm_i ] Phi, Psi_i(0) = 0         (6)

``dA/dm_i`` is (2) differentiated in the mass coefficients:

    dA/dm1: [u1,q1] += -Dg(q1);  [u2,q1] += Dg(d) - Dg(q1);  [u2,q2] += -Dg(d)
    dA/dm2: [u1,q1] += -Dg(d);   [u1,q2] += Dg(d) - Dg(q2);  [u2,q2] += -Dg(q2)

``D_z A . S`` is (2) with every ``Dg(x)`` replaced by the directional second
derivative ``T(x, s) = D^2 g(x) . s`` evaluated along the matching displacement
(``s = S_q1`` for ``Dg(q1)``, ``s = S_q2`` for ``Dg(q2)``, ``s = S_q2 - S_q1``
for ``Dg(d)``), with

    T(x,s)_{ab} = -3 ( delta_ab (x.s) + s_a x_b + x_a s_b ) / |x|^5
                  + 15 x_a x_b (x.s) / |x|^7                          (7)

Closing the loop: the corrected family
--------------------------------------
The verifier's ``dG-/dm`` is a *total* derivative along the corrected periodic
family, i.e. ``p`` is re-solved at every mass.  The closure residual is

    F(p, m) = phi_T( c(p, m), m ) - c(p, m)   in R^8, with p in R^4.

It is overdetermined but consistent (four of the eight conditions are implied by
the reversible symmetry of the chart and by the first integrals), and both the
Julia and Python correctors solve it in the Gauss--Newton least-squares sense.
Implicit differentiation therefore gives

    dp/dm = -J_p^+ (dF/dm),                                            (8)
    J_p[:, 0:3] = (M - I) C,   C = dc/d(x1,v1,v2),   J_p[:, 3] = f(z(T))
    dF/dm_i = S_i^partial(T) - dc/dm_i,     S_i^partial(0) = dc/dm_i

The least-squares residual of (8) is returned as ``dp_lstsq_residual``; if the
derivation were wrong (or the family were not locally unique) it would not be at
round-off.  That is a live consistency check, not a decoration.

The monodromy used by the event functions is ``M(m) = Phi(T(m); c(p(m),m), m)``,
so its total mass derivative picks up three terms -- explicit, initial-condition,
and period:

    dM/dm_i = Psi_i^total(T) + A(z(T)) M . dT/dm_i                     (9)

where ``Psi_i^total`` is (6) integrated with the *total* initial condition
``S_i(0) = C (dp/dm_i)[0:3] + dc/dm_i``.  (``dT/dm_i = (dp/dm_i)[3]``; the period
does not appear in ``c``, so it only enters through (9).)

Event derivatives
-----------------
With ``alpha = tr M``, ``beta = (alpha^2 - tr M^2)/2``:

    d alpha = tr(dM),   d beta = tr( (alpha I - M) dM )

hence every event derivative is a single trace against a weight matrix:

    G+    = beta - 6 alpha + 20   ->  W = (alpha - 6) I - M
    G-    = beta - 2 alpha +  4   ->  W = (alpha - 2) I - M           (10)
    Delta = (alpha-4)^2 - 4(beta - 4 alpha + 8)
                                  ->  W = (8 - 2 alpha) I + 4 M
    dEvent/dm_i = tr( W . dM/dm_i )

Cost
----
Per mass pair the sensitivity path is two integrations over one period:

    pass 1:  8 + 64 + 16               =  88 components
    pass 2:  8 + 64 + 16 + 128         = 216 components

and it returns dG/dm1 AND dG/dm2 together, for all three event modes, with no
step size.  The stencil path it replaces is 2 stencils x 4 nodes x (four Newton
iterations) x 72 components, serialised.

Measured on the fold-root orbit (period 7.4553, DOP853 rtol 1e-13/atol 1e-15,
best of three): one chart correction 2.44 s; one ``mass_sensitivity`` call
4.39 s (1.80 corrections); one five-point mass stencil 11.78 s (4.83
corrections).  2.7x per scalar derivative, 5.4x for the pair ``audit_fold``
needs at ONE step size, ~15x once its four-step-size ladder is deleted -- the
ladder only exists because the derivative was approximate.  Full accounting and
the Julia port plan: ``research/MASS_SENSITIVITY_PORT_PLAN.md``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.integrate import solve_ivp

Array = np.ndarray
EventMode = Literal["plus_one", "minus_one", "trace_collision"]
EVENT_MODES: tuple[EventMode, ...] = ("plus_one", "minus_one", "trace_collision")

_I2 = np.eye(2)


# --------------------------------------------------------------------------- #
# Newtonian kernel and its first two derivatives
# --------------------------------------------------------------------------- #
def kernel(x: Array) -> tuple[Array, Array]:
    """Return ``g(x) = x/|x|^3`` and ``Dg(x)``.

    Written with ``x @ x`` and ``sqrt`` only (no ``abs``/``norm``) so the whole
    stack stays complex-analytic and admits complex-step differentiation.
    """
    r2 = x @ x
    if r2 == 0:
        raise FloatingPointError("binary collision")
    r = np.sqrt(r2)
    i3 = 1.0 / (r * r2)
    i5 = i3 / r2
    return x * i3, _I2 * i3 - 3.0 * np.outer(x, x) * i5


def kernel_hessian_contract(x: Array, s: Array) -> Array:
    """Return ``T(x, s) = D^2 g(x) . s`` -- equation (7) of the module docstring."""
    r2 = x @ x
    if r2 == 0:
        raise FloatingPointError("binary collision")
    r = np.sqrt(r2)
    i5 = 1.0 / (r * r2 * r2)
    i7 = i5 / r2
    xs = x @ s
    return -3.0 * (_I2 * xs + np.outer(s, x) + np.outer(x, s)) * i5 + 15.0 * np.outer(x, x) * xs * i7


# --------------------------------------------------------------------------- #
# Reduced field, state Jacobian, mass partials
# --------------------------------------------------------------------------- #
def _blocks(z: Array) -> tuple[Array, Array, Array, Array, Array, Array]:
    q1, q2 = z[0:2], z[2:4]
    d = q2 - q1
    g1, dg1 = kernel(q1)
    g2, dg2 = kernel(q2)
    gd, dgd = kernel(d)
    return g1, g2, gd, dg1, dg2, dgd


def _assemble_jacobian(masses, dg1: Array, dg2: Array, dgd: Array, *, dtype) -> Array:
    """Fill the (2) block pattern; also used with ``Dg -> T`` for ``D_z A . S``."""
    m1, m2, m3 = masses
    a = np.zeros((8, 8), dtype=dtype)
    a[4:6, 0:2] = -m2 * dgd - (m1 + m3) * dg1
    a[4:6, 2:4] = m2 * dgd - m2 * dg2
    a[6:8, 0:2] = m1 * dgd - m1 * dg1
    a[6:8, 2:4] = -m1 * dgd - (m2 + m3) * dg2
    return a


def reduced_field(z: Array, masses) -> tuple[Array, Array]:
    """Return ``f(z, m)`` and ``A = D_z f`` for the 8D reduced system."""
    z = np.asarray(z)
    dtype = np.result_type(z.dtype, np.asarray(masses).dtype)
    g1, g2, gd, dg1, dg2, dgd = _blocks(z)
    m1, m2, m3 = masses
    f = np.zeros(8, dtype=dtype)
    f[0:4] = z[4:8]
    f[4:6] = m2 * gd - (m1 + m3) * g1 - m2 * g2
    f[6:8] = -m1 * gd - m1 * g1 - (m2 + m3) * g2
    a = _assemble_jacobian(masses, dg1, dg2, dgd, dtype=dtype)
    a[0:4, 4:8] = np.eye(4)
    return f, a


def field_mass_partial(z: Array, masses) -> Array:
    """Return ``df/dm`` as an 8x2 matrix (columns m1, m2; m3 fixed) -- eq. (3)."""
    z = np.asarray(z)
    dtype = np.result_type(z.dtype, np.asarray(masses).dtype)
    g1, g2, gd, _, _, _ = _blocks(z)
    out = np.zeros((8, 2), dtype=dtype)
    out[4:6, 0] = -g1
    out[6:8, 0] = -gd - g1
    out[4:6, 1] = gd - g2
    out[6:8, 1] = -g2
    return out


def jacobian_mass_partial(z: Array, masses) -> Array:
    """Return ``dA/dm`` with shape (2, 8, 8) (index 0 -> m1, index 1 -> m2)."""
    z = np.asarray(z)
    dtype = np.result_type(z.dtype, np.asarray(masses).dtype)
    _, _, _, dg1, dg2, dgd = _blocks(z)
    out = np.zeros((2, 8, 8), dtype=dtype)
    out[0, 4:6, 0:2] = -dg1
    out[0, 6:8, 0:2] = dgd - dg1
    out[0, 6:8, 2:4] = -dgd
    out[1, 4:6, 0:2] = -dgd
    out[1, 4:6, 2:4] = dgd - dg2
    out[1, 6:8, 2:4] = -dg2
    return out


def jacobian_state_directional(z: Array, masses, s: Array) -> Array:
    """Return ``D_z A . s``: the (2) pattern with ``Dg(x) -> T(x, ds)``."""
    z = np.asarray(z)
    s = np.asarray(s)
    dtype = np.result_type(z.dtype, s.dtype, np.asarray(masses).dtype)
    q1, q2 = z[0:2], z[2:4]
    su, sw = s[0:2], s[2:4]
    t1 = kernel_hessian_contract(q1, su)
    t2 = kernel_hessian_contract(q2, sw)
    td = kernel_hessian_contract(q2 - q1, sw - su)
    return _assemble_jacobian(masses, t1, t2, td, dtype=dtype)


# --------------------------------------------------------------------------- #
# Shooting chart
# --------------------------------------------------------------------------- #
def chart_state(p, masses) -> Array:
    """Return ``c(p, m)``: the Li--Li--Liao chart point in reduced coordinates."""
    m1, m2, m3 = masses
    x1, v1, v2 = p[0], p[1], p[2]
    dtype = np.result_type(np.asarray(p).dtype, np.asarray(masses).dtype)
    v3 = -(m1 * v1 + m2 * v2) / m3
    z = np.zeros(8, dtype=dtype)
    z[0] = x1
    z[2] = 1.0
    z[5] = v1 - v3
    z[7] = v2 - v3
    return z


def chart_param_tangent(masses) -> Array:
    """Return ``C = dc/d(x1, v1, v2)`` as 8x3."""
    m1, m2, m3 = masses
    c = np.zeros((8, 3), dtype=np.result_type(np.asarray(masses).dtype, float))
    c[0, 0] = 1.0
    c[5, 1] = 1.0 + m1 / m3
    c[7, 1] = m1 / m3
    c[5, 2] = m2 / m3
    c[7, 2] = 1.0 + m2 / m3
    return c


def chart_mass_tangent(p, masses) -> Array:
    """Return ``dc/dm`` as 8x2 -- equation (4).  Zero only if v1 = v2 = 0."""
    _, _, m3 = masses
    v1, v2 = p[1], p[2]
    dtype = np.result_type(np.asarray(p).dtype, np.asarray(masses).dtype)
    b = np.zeros((8, 2), dtype=dtype)
    b[5, 0] = v1 / m3
    b[7, 0] = v1 / m3
    b[5, 1] = v2 / m3
    b[7, 1] = v2 / m3
    return b


# --------------------------------------------------------------------------- #
# Event algebra
# --------------------------------------------------------------------------- #
def event_from_monodromy(monodromy: Array, mode: EventMode):
    a = np.asarray(monodromy)
    alpha = np.trace(a)
    beta = 0.5 * (alpha * alpha - np.trace(a @ a))
    if mode == "plus_one":
        return beta - 6.0 * alpha + 20.0
    if mode == "minus_one":
        return beta - 2.0 * alpha + 4.0
    if mode == "trace_collision":
        return (alpha - 4.0) ** 2 - 4.0 * (beta - 4.0 * alpha + 8.0)
    raise ValueError(f"unsupported event mode: {mode}")


def event_weight(monodromy: Array, mode: EventMode) -> Array:
    """Return ``W`` with ``dEvent = tr(W dM)`` -- equation (10)."""
    a = np.asarray(monodromy)
    alpha = np.trace(a)
    eye = np.eye(8, dtype=a.dtype)
    if mode == "plus_one":
        return (alpha - 6.0) * eye - a
    if mode == "minus_one":
        return (alpha - 2.0) * eye - a
    if mode == "trace_collision":
        return (8.0 - 2.0 * alpha) * eye + 4.0 * a
    raise ValueError(f"unsupported event mode: {mode}")


# --------------------------------------------------------------------------- #
# Augmented integration
# --------------------------------------------------------------------------- #
#: Packed layout: z[0:8], Phi[8:72], S[72:88] as (2, 8), Psi[88:216] as (2, 8, 8).
_N_PSI = 128


def _augmented_rhs(_t, y, masses, level: int) -> Array:
    """``level`` 0: z+Phi.  1: z+Phi+S.  2: z+Phi+S+Psi."""
    z = y[0:8]
    phi = y[8:72].reshape(8, 8)
    f, a = reduced_field(z, masses)
    dphi = a @ phi
    if level == 0:
        return np.concatenate((f, dphi.ravel()))
    s = y[72:88].reshape(2, 8)
    ds = s @ a.T + field_mass_partial(z, masses).T
    if level == 1:
        return np.concatenate((f, dphi.ravel(), ds.ravel()))
    psi = y[88:216].reshape(2, 8, 8)
    dadm = jacobian_mass_partial(z, masses)
    dpsi = np.empty_like(psi)
    for i in range(2):
        dpsi[i] = a @ psi[i] + (jacobian_state_directional(z, masses, s[i]) + dadm[i]) @ phi
    return np.concatenate((f, dphi.ravel(), ds.ravel(), dpsi.ravel()))


def _integrate(z0, s0, period, masses, *, level, rtol, atol):
    """Integrate ``z, Phi`` (and optionally ``S``, ``Psi``) over one period."""
    parts = [np.asarray(z0, dtype=float), np.eye(8).ravel()]
    if level >= 1:
        parts.append(np.asarray(s0, dtype=float).ravel())
    if level >= 2:
        parts.append(np.zeros(_N_PSI))
    y0 = np.concatenate(parts)
    sol = solve_ivp(
        lambda t, y: _augmented_rhs(t, y, masses, level),
        (0.0, float(period)),
        y0,
        method="DOP853",
        rtol=rtol,
        atol=atol,
    )
    if not sol.success:
        raise RuntimeError(sol.message)
    yf = sol.y[:, -1]
    z_t = yf[0:8]
    phi_t = yf[8:72].reshape(8, 8)
    s_t = yf[72:88].reshape(2, 8) if level >= 1 else None
    psi_t = yf[88:216].reshape(2, 8, 8) if level >= 2 else None
    return z_t, phi_t, s_t, psi_t, int(sol.nfev)


# --------------------------------------------------------------------------- #
# Public result + driver
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class MassSensitivity:
    masses: tuple[float, float, float]
    p: tuple[float, float, float, float]
    closure_norm: float
    monodromy: Array
    alpha: float
    beta: float
    events: dict[str, float]
    #: total dEvent/dm along the corrected family, keys are event modes, 2-vectors
    d_events_dm: dict[str, Array]
    #: the same, with the period held frozen (i.e. omitting the dT/dm term)
    d_events_dm_fixed_period: dict[str, Array]
    dp_dm: Array          # 4x2, d(x1,v1,v2,T)/d(m1,m2)
    dm_monodromy: Array   # 2x8x8, total dM/dm
    dp_lstsq_residual: float
    #: the same residual divided by ||dF/dm||.  The 8x4 closure system is
    #: consistent, so this must sit at round-off; if it does not, either the
    #: derivation is wrong or the periodic family is not locally unique here.
    dp_lstsq_relative_residual: float
    #: singular values of the 8x4 closure Jacobian.  A collapsing smallest value
    #: means dp/dm is meaningless no matter how exact the sensitivities are.
    closure_jacobian_singular_values: Array
    rhs_evaluations: int


def mass_sensitivity(
    p,
    masses,
    *,
    rtol: float = 1e-12,
    atol: float = 1e-14,
) -> MassSensitivity:
    """Return exact (to integrator tolerance) mass sensitivities of the events.

    ``p`` must already be a corrected chart point; the sensitivities are the
    total derivatives along the corrected periodic family, i.e. the same
    quantity the five-point stencils in ``verify_secondary_minus_fold.jl``
    approximate.
    """
    p = np.asarray(p, dtype=float)
    masses = tuple(float(m) for m in masses)
    period = float(p[3])
    if period <= 0.0:
        raise ValueError("period must be positive")

    z0 = chart_state(p, masses)
    dcdm = chart_mass_tangent(p, masses)
    cmat = chart_param_tangent(masses)

    # Pass 1: partial sensitivities at frozen p -> dF/dm -> dp/dm.  Equation (8).
    z_t, mono, s_part, _, nfev1 = _integrate(
        z0, dcdm.T, period, masses, level=1, rtol=rtol, atol=atol
    )
    dfdm_closure = s_part.T - dcdm                     # 8x2
    f_end, a_end = reduced_field(z_t, masses)
    jp = np.empty((8, 4))
    jp[:, 0:3] = (mono - np.eye(8)) @ cmat
    jp[:, 3] = f_end
    dpdm, *_ = np.linalg.lstsq(jp, -dfdm_closure, rcond=None)
    lstsq_residual = float(np.linalg.norm(jp @ dpdm + dfdm_closure))
    dfdm_scale = float(np.linalg.norm(dfdm_closure))
    lstsq_relative = lstsq_residual / dfdm_scale if dfdm_scale > 0.0 else lstsq_residual
    singular_values = np.linalg.svd(jp, compute_uv=False)

    # Pass 2: total sensitivities and dPhi/dm.  Equations (5), (6), (9).
    s0_total = dcdm + cmat @ dpdm[0:3, :]
    _, mono2, _, psi_t, nfev2 = _integrate(
        z0, s0_total.T, period, masses, level=2, rtol=rtol, atol=atol
    )
    period_term = a_end @ mono2
    dm_mono = np.empty((2, 8, 8))
    for i in range(2):
        dm_mono[i] = psi_t[i] + period_term * dpdm[3, i]

    events, d_events, d_events_fixed_t = {}, {}, {}
    for mode in EVENT_MODES:
        w = event_weight(mono2, mode)
        events[mode] = float(event_from_monodromy(mono2, mode))
        d_events[mode] = np.array([float(np.trace(w @ dm_mono[i])) for i in range(2)])
        d_events_fixed_t[mode] = np.array([float(np.trace(w @ psi_t[i])) for i in range(2)])

    alpha = float(np.trace(mono2))
    beta = float(0.5 * (alpha * alpha - np.trace(mono2 @ mono2)))
    return MassSensitivity(
        masses=masses,
        p=(float(p[0]), float(p[1]), float(p[2]), float(p[3])),
        closure_norm=float(np.linalg.norm(z_t - z0)),
        monodromy=mono2,
        alpha=alpha,
        beta=beta,
        events=events,
        d_events_dm=d_events,
        d_events_dm_fixed_period=d_events_fixed_t,
        dp_dm=dpdm,
        dm_monodromy=dm_mono,
        dp_lstsq_residual=lstsq_residual,
        dp_lstsq_relative_residual=lstsq_relative,
        closure_jacobian_singular_values=singular_values,
        rhs_evaluations=nfev1 + nfev2,
    )


def integrate_frozen_chart(p, masses, *, level: int = 2, rtol: float = 1e-12, atol: float = 1e-14):
    """Integrate ``z, Phi[, S, Psi]`` from the chart point with ``p`` held fixed.

    Returns ``(z_T, Phi_T, S_T, Psi_T, nfev)``; ``S`` starts from ``dc/dm`` so the
    returned sensitivities are the frozen-chart ones.
    """
    z0 = chart_state(p, masses)
    s0 = chart_mass_tangent(p, masses).T
    return _integrate(z0, s0, float(p[3]), masses, level=level, rtol=rtol, atol=atol)


def frozen_chart_mass_derivative(
    p,
    masses,
    *,
    rtol: float = 1e-12,
    atol: float = 1e-14,
) -> tuple[dict[str, Array], Array]:
    """Return ``dEvent/dm`` and ``dM/dm`` with the chart parameters ``p`` FROZEN.

    This is not the quantity the fold verifier wants (that one follows the
    corrected family), but it is the piece that can be checked to machine
    precision by a complex step, because no implicit solve intervenes.  See
    ``tests/test_mass_sensitivity.py``.
    """
    p = np.asarray(p, dtype=float)
    masses = tuple(float(m) for m in masses)
    z0 = chart_state(p, masses)
    dcdm = chart_mass_tangent(p, masses)
    _, mono, _, psi_t, _ = _integrate(
        z0, dcdm.T, float(p[3]), masses, level=2, rtol=rtol, atol=atol
    )
    out = {}
    for mode in EVENT_MODES:
        w = event_weight(mono, mode)
        out[mode] = np.array([float(np.trace(w @ psi_t[i])) for i in range(2)])
    return out, psi_t


def flow_and_monodromy(p, masses, *, rtol: float = 1e-12, atol: float = 1e-14):
    """Integrate ``z`` and ``Phi`` over one period; complex-``masses`` safe.

    Used by the complex-step validation: ``masses`` may carry an imaginary
    perturbation, in which case the returned monodromy is complex and its
    imaginary part divided by the step is the exact frozen-chart mass
    derivative.
    """
    masses = tuple(masses)
    z0 = chart_state(p, masses)
    y0 = np.concatenate((z0, np.eye(8, dtype=z0.dtype).ravel()))
    sol = solve_ivp(
        lambda t, y: _augmented_rhs(t, y, masses, 0),
        (0.0, float(np.real(p[3]))),
        y0,
        method="DOP853",
        rtol=rtol,
        atol=atol,
    )
    if not sol.success:
        raise RuntimeError(sol.message)
    yf = sol.y[:, -1]
    return yf[0:8], yf[8:72].reshape(8, 8)


# --------------------------------------------------------------------------- #
# Reference path: the corrected sample the stencils differentiate
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class CorrectedSample:
    p: Array
    masses: tuple[float, float, float]
    monodromy: Array
    closure_norm: float
    iterations: int

    def event(self, mode: EventMode) -> float:
        return float(event_from_monodromy(self.monodromy, mode))


def corrected_sample(
    masses,
    guess,
    *,
    target: float = 5e-14,
    max_closure: float = 1e-9,
    rtol: float = 1e-13,
    atol: float = 1e-15,
    maxiter: int = 20,
) -> CorrectedSample:
    """Gauss--Newton chart correction mirroring ``correct_chart`` in Julia.

    This is the float64 stand-in for ``corrected_at``; it exists so the
    sensitivity path can be validated against the very finite differences it is
    meant to replace.

    ``target`` is the closure the Newton aims for; float64 typically stalls near
    1e-12 instead, so a stalled iterate is accepted only if it is still below
    ``max_closure`` (1e-9 by default, two orders stricter than the project's
    frozen 1e-7 closure gate).  A stall above that RAISES: a silently
    unconverged node poisons a finite-difference stencil without any visible
    symptom, which is precisely how a large-``h`` stencil can report a wildly
    wrong derivative.  ``julia/verify_secondary_minus_fold.jl`` is safe here
    because ``correct_chart`` errors out when it misses its closure target.
    """
    masses = tuple(float(m) for m in masses)
    p = np.asarray(guess, dtype=float).copy()
    cmat = chart_param_tangent(masses)
    last = np.inf
    last_state: tuple[Array, Array, float] | None = None
    for it in range(1, maxiter + 1):
        z0 = chart_state(p, masses)
        z_t, mono, _, _, _ = _integrate(
            z0, None, p[3], masses, level=0, rtol=rtol, atol=atol
        )
        residual = z_t - z0
        rn = float(np.linalg.norm(residual))
        if rn <= target:
            return CorrectedSample(p.copy(), masses, mono, rn, it)
        if rn >= last:
            if last_state is not None and last <= max_closure:
                p_prev, mono_prev, _ = last_state
                return CorrectedSample(p_prev, masses, mono_prev, last, it)
            raise RuntimeError(
                f"chart correction stalled at closure {min(rn, last):.3e} "
                f"(> max_closure {max_closure:.1e}) for masses {masses}"
            )
        last = rn
        last_state = (p.copy(), mono, rn)
        jac = np.empty((8, 4))
        jac[:, 0:3] = (mono - np.eye(8)) @ cmat
        jac[:, 3] = reduced_field(z_t, masses)[0]
        delta, *_ = np.linalg.lstsq(jac, -residual, rcond=None)
        p = p + delta
        if p[3] <= 0.0:
            raise RuntimeError("shooting produced non-positive period")
    if last <= max_closure and last_state is not None:
        return CorrectedSample(last_state[0], masses, last_state[1], last, maxiter)
    raise RuntimeError(f"chart correction did not converge; closure {last:.3e}")


def finite_difference_event_derivative(
    masses,
    p,
    mode: EventMode,
    axis: int,
    h: float,
    *,
    target: float = 5e-14,
    rtol: float = 1e-13,
    atol: float = 1e-15,
) -> float:
    """Central difference of the *corrected* event -- one stencil, two nodes."""
    out = []
    for sign in (+1.0, -1.0):
        m = list(masses)
        m[axis] += sign * h
        out.append(corrected_sample(m, p, target=target, rtol=rtol, atol=atol).event(mode))
    return (out[0] - out[1]) / (2.0 * h)


def second_mass_derivative(
    masses,
    p,
    mode: EventMode,
    axis: int,
    h: float,
    *,
    rtol: float = 1e-13,
    atol: float = 1e-15,
) -> tuple[Array, float]:
    """Second mass derivative as ONE difference of exact first derivatives.

    ``audit_fold`` needs ``d2G/dm2^2`` (for the fold curvature) as well as
    ``dG/dm2`` and ``dG/dm1``.  Because the first derivatives here are exact, the
    second derivative is a *first* difference of them rather than a second
    difference of function values: two corrected nodes instead of four, and the
    step-size error drops from ``O(h^2) + noise/h^2`` to ``O(h^2) + noise/h``.

    Returns ``(d/dm_axis of the full dEvent/dm gradient, that gradient's own
    component along ``axis``)`` -- i.e. a row of the mass Hessian plus the pure
    second derivative.
    """
    grads = {}
    for sign in (+1.0, -1.0):
        m = list(masses)
        m[axis] += sign * h
        node = corrected_sample(m, p, rtol=rtol, atol=atol)
        grads[sign] = mass_sensitivity(node.p, m, rtol=rtol, atol=atol).d_events_dm[mode]
    row = (grads[+1.0] - grads[-1.0]) / (2.0 * h)
    return row, float(row[axis])


def richardson_event_derivative(
    masses,
    p,
    mode: EventMode,
    axis: int,
    h: float,
    **kwargs,
) -> tuple[float, float, float]:
    """Return ``(richardson, d(h), d(h/2))`` for the corrected-event derivative.

    The Richardson combination ``(4 d(h/2) - d(h)) / 3`` removes the ``O(h^2)``
    truncation term of the central difference, leaving ``O(h^4)`` truncation plus
    amplified round-off -- the best a float64 stencil can do.
    """
    d1 = finite_difference_event_derivative(masses, p, mode, axis, h, **kwargs)
    d2 = finite_difference_event_derivative(masses, p, mode, axis, h / 2.0, **kwargs)
    return (4.0 * d2 - d1) / 3.0, d1, d2
