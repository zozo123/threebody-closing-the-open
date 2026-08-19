# Post drafts for X

Written to be accurate. The result is genuinely open, so nothing here claims otherwise. Pick one.

## Short version

New preprint. The Li--Li--Liao catalog of 135,445 three-body orbits was read in 2023 as two
families. It is one: the plot that splits it folds over on itself, so two visible branches are one
sheet seen twice. We locate two stability transitions to a part in 10^9 and name their mechanisms.

Status: open. [link]

## The version I would actually post

We set out to map where a family of three-body orbits loses stability, and found something about
our own tools instead.

The accuracy gate our pipeline enforces (2e-8) is finer than the accuracy the pipeline can
reproduce. Rotate the coordinate frame, redo the same computation at the same tolerance, and the
answer moves by more than the gate on 345 of 620 cases.

Recompute three points at 60 digits, at the exact masses the fast arithmetic reported: the fast
version said all three passed comfortably. The careful version put two of them 56x outside the gate,
with the real answers a small distance away.

And one of our own pass/fail conditions turns out to be unsatisfiable. It asks for complete coverage
of a parameter region in which the orbit family does not exist. You cannot survey what is not there.

What we can say: the catalog is one continuation-connected family, not two, because the projection
that appeared to split it is folded and non-injective. Two transitions are bracketed to a part in
10^9 with their mechanisms named.

What we cannot: the critical graph is not closed. Two ends stay unclassified. Status: open, and the
abstract says so.

Every number maps to a committed artifact and an evidence rung. [link]

## Thread version

1/ The Li--Li--Liao table lists 135,445 unequal-mass three-body periodic orbits and marks which are
stable. A 2023 analysis of the same table read it as two families. We asked whether that is real.

2/ It is not two families. The plot that separates them uses two summary invariants, and on this
sheet that projection folds over on itself: 590 places where it flips, and 86 far-separated orbit
pairs landing nearly on top of each other. Two branches in a shadow, one object.

3/ We then located two points where stability changes, to about one part in 10^9, and identified the
mechanism at each: an eigenvalue crossing +1, and two eigenvalue pairs colliding on the unit circle
with opposite Krein sign. The signature, not the eigenvalue positions, is what names the second.

4/ The uncomfortable part. Our accuracy gate is 2e-8. Rotating the coordinate frame and redoing the
same computation at the same tolerance moves the event by more than that gate on 345 of 620 cases,
and by ten times it on 144. The gate was partly measuring the coordinate frame.

5/ Worse, passing the gate does not mean the answer is in the right place. Three points that the
fast arithmetic reported comfortably inside it were, at 60 digits and at the same masses, outside by
up to 56x, with the true roots a small distance away.

6/ And a release condition in our own pipeline is unsatisfiable, not merely unmet: it demands
converged coverage of a region where no orbit of the family closes. ~97% of the failed probes there
have residuals 7 orders outside the gate. There is nothing to survey.

7/ So: one family, two named and bracketed transitions, and a set of results about how this class of
computation should be certified. The critical graph is not closed, two ends stay unclassified, and
the status is open. The abstract says so. [link]

## Notes before posting

- Replace [link] with the arXiv abs URL once it is live, not the PDF.
- Do not add "solved", "breakthrough", or "first". No priority claim is made in the paper and one
  should not be made here.
- If a reply asks whether this solves the three-body problem: no, and the paper's first sentence
  says the general problem is not solved and nothing in it bears on that.
- 345/620, 144/620, 56x, 590, 86, 135,445 and the 10^9 bracket widths are all in the paper. The 97%
  figure is the fraction of failing probes with closure above 1e-2 in the two committed audits.
