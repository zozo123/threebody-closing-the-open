# Universal Floquet event geometry for the COM-reduced family

## Purpose

This note isolates the algebra that is independent of the particular periodic-orbit family.  It turns the numerical stability map in mass space into the preimage of a universal spectral-stability domain in two trace invariants.

This is structural mathematics, not a discovery claim about the Li--Li--Liao family by itself.

## Reduced trace polynomial

For the 8-dimensional center-of-mass-reduced monodromy matrix, four Floquet multipliers are the trivial/symmetry unit multipliers.  Let the remaining four multipliers form two reciprocal pairs

\[
(\lambda_1,\lambda_1^{-1}),\qquad(\lambda_2,\lambda_2^{-1}).
\]

Define

\[
t_i = \lambda_i+\lambda_i^{-1}.
\]

With

\[
\alpha=\operatorname{tr}M,\qquad
\beta=\frac{(\operatorname{tr}M)^2-\operatorname{tr}(M^2)}{2},
\]

the two nontrivial trace roots satisfy

\[
P(t)=t^2-(\alpha-4)t+\beta-4\alpha+8=0.
\]

Hence

\[
t_1+t_2=\alpha-4,\qquad
t_1t_2=\beta-4\alpha+8.
\]

## Spectral stability domain

For a reciprocal pair, both multipliers lie on the unit circle exactly when its real trace root satisfies

\[
-2\le t\le2.
\]

Thus the four nontrivial multipliers are spectrally stable exactly when both roots of `P` are real and lie in `[-2,2]` (with boundary degeneracies treated separately).

The three algebraic boundary equations are therefore

### Pair at +1

\[
P(2)=\beta-6\alpha+20=0.
\]

### Pair at -1

\[
P(-2)=\beta-2\alpha+4=0.
\]

### Collision of the two trace roots

\[
\Delta=(\alpha-4)^2-4(\beta-4\alpha+8)
       =\alpha^2+8\alpha-16-4\beta=0.
\]

The stable region in the `(alpha,beta)` plane is bounded by these three pieces.  Its three spectral vertices are exact:

| trace roots | multipliers | `(alpha,beta)` |
|---|---|---|
| `(-2,-2)` | both nontrivial pairs at `-1` | `(0,-4)` |
| `(-2,+2)` | one pair at `-1`, one at `+1` | `(4,4)` |
| `(+2,+2)` | both nontrivial pairs at `+1` | `(8,28)` |

The mixed vertex `(4,4)` is especially important because a stability-boundary component can change its active mechanism from `+1` to `-1` only by passing through this point, unless a coarse parameter cell contains several distinct critical roots and the apparent switch is an aliasing artifact.

## Mass-space interpretation

Let

\[
\Phi:(m_1,m_2)\mapsto(\alpha(m_1,m_2),\beta(m_1,m_2))
\]

be the trace-invariant map restricted to one continuation-connected periodic-orbit sheet (with `m3=1`).

Then the linear-stability set in mass space is

\[
\Phi^{-1}(\mathcal S),
\]

where `S` is the universal stable trace-root domain above, and the mass-space critical network is the preimage of its boundary.

This gives a useful taxonomy of observed geometry:

1. **ordinary critical branch:** regular preimage of one boundary equation;
2. **mass-plane fold:** one event equation remains active but its preimage is tangent to a chosen mass slicing, creating/annihilating two crossings without changing spectral mechanism;
3. **mixed event vertex:** preimage of `(alpha,beta)=(4,4)`, where `+1` and `-1` branches meet;
4. **double +1 or double -1 vertex:** preimages of `(8,28)` or `(0,-4)`;
5. **Krein/Hamiltonian--Hopf boundary:** the `Delta=0` branch when two simple unit-circle modes of opposite Krein signature collide and subsequently form a reciprocal complex quartet.  The name requires canonical symplectic/Krein evidence, not merely `Delta=0` in noncanonical coordinates.

## Current numerical implications

The coarse 620-transition audit on the frozen Li--Li--Liao table already falsifies a two-curve description of the stability boundary:

- the principal upper `S->U` track is a trace-collision branch at all sampled representatives;
- the secondary upper track is a `-1` branch at all sampled representatives;
- the secondary lower track changes from `-1` to `+1`;
- the principal lower track changes `+1 -> -1 -> +1`.

Therefore the publishable object to compute is a **Floquet event network** in mass space: smooth event arcs, folds of those arcs, and exact codimension-two vertex preimages.  A rectangular S/U grid is only a coarse sampling of that network.

## Claim discipline

A float64 event localization is a screening candidate.  Event vertices and representative branches enter a release claim only after:

1. periodic closure convergence;
2. independent BigFloat reproduction;
3. canonical symplectic validation for mechanism labels that require it;
4. cross-implementation agreement on branch/event identity;
5. adversarial checks for hidden multiple zeros inside coarse cells.
