This repository contains the source of the **O5** Cosmochrony paper  
[*Subdiffusive Valence Growth under Bounded Relational Flux: Structural Derivation of the Cascade Exponent*](out/SpectralO5.pdf).

This work extends the **spectral relaxation programme** by addressing the final
remaining structural parameter left undetermined by O4: the cascade exponent
$\beta$ governing the growth law

$p(n) \sim n^\beta$.

While **O4** closes the mass spectrum through a unified stabilisation mechanism
for all three ADE levels, it still treats $\beta$ as an external structural input.
The present work removes that freedom at the level of admissible growth laws by
showing that bounded relational flux imposes a non-trivial constraint on the
allowed dynamics of valence growth.

The central result is that the Born--Infeld saturation bound inherited from the
substrate,

$|\partial_t \chi_v| \le c_\chi$,

excludes all superlinear growth laws and implies the structural bound

$\beta \le 1$.

# Core Result

The paper establishes a **first-principles upper bound** on the cascade exponent.

Starting from:

- the bounded-flux constraint on the substrate
- the Ramanujan--LPS relaxation graph model
- a closure hypothesis relating effective valence to cumulative exploration of
  the relational configuration space

the analysis shows that the growth of valence satisfies

$\Delta p(n) \lesssim c_\chi \sqrt{p(n)}$,

which integrates to

$p(n) \lesssim \frac{1}{4} c_\chi^2 n^2$.

As a consequence, any admissible power-law ansatz

$p(n) \sim n^\beta$

must satisfy

$\beta \le 1$.

# Structural Role of O5

O5 does not yet derive the exact value of $\beta$, but it removes an entire class
of otherwise possible growth laws.

This resolves the last major arbitrariness in the O-series at the level of
scaling structure:

- **O3** showed that $\beta$ controls the amplification of the mass hierarchy
- **O4** closed the full three-generation spectrum once $\beta$ is given
- **O5** proves that $\beta$ cannot be superlinear

The cascade exponent is therefore no longer an unconstrained positive parameter,
but a structurally restricted quantity inherited from bounded relational flux.

# What O5 Adds

O5 introduces a new layer of structural necessity:

- the valence growth law is not free
- the bounded-flux condition constrains the rate at which new relational
  structure can become accessible
- the mass-hierarchy mechanism of O3--O4 must therefore operate within a
  sublinear or at most linear regime

This significantly strengthens the internal consistency of the programme.

# Interpretation of the Bound

The inequality

$\beta \le 1$

has a clear physical meaning in the Cosmochrony framework.

It states that the relational substrate cannot explore new effective valence
faster than allowed by its finite flux capacity.
The relaxation front may widen along the cascade, but not arbitrarily fast.
Superlinear growth would require an effective propagation of relational
accessibility incompatible with the saturation bound.

Thus the observed hierarchy cannot arise from unrestricted graph expansion,
but only from a dynamically constrained exploration process.

# Relation to Previous Steps

O5 preserves all previous structural results:

- spectral admissibility from **Step 1**
- binary-polyhedral maximality from **Step 2**
- three-level ADE stratigraphy from **Step 3**
- projective dynamics and support contraction from **O1**
- hierarchical amplification via growing valence from **O3**
- unified mass closure from **O4**

It does not modify the mass formula itself.
Instead, it constrains the only remaining dynamical degree of freedom entering
that formula.

# Conceptual Structure

O5 completes the next stage of the structural chain:

1. Spectral admissibility → mode selection
2. Spectral capacity → binary-polyhedral maximality
3. Spectral stratigraphy → discrete ADE levels
4. O1 → ordering via support contraction
5. O3 → amplification via valence growth
6. O4 → closure of the full mass spectrum
7. O5 → structural bound on the growth exponent $\beta$

The programme now contains no unconstrained superlinear cascade law.

# What O5 Resolves

O5 provides:

- a structural derivation of the upper bound $\beta \le 1$
- a first-principles exclusion of superlinear valence growth
- a direct link between bounded Born--Infeld flux and cascade kinematics
- a substantial reduction in the residual arbitrariness of the O-series

It shows that the mass hierarchy mechanism is not merely compatible with bounded
relational dynamics, but is directly constrained by it.

# Residual Open Problem

What remains open is no longer the qualitative class of $\beta$, but its
**quantitative value**.

O5 does not yet derive the phenomenologically relevant window for $\beta$.
It only proves that the exact exponent must lie within the structurally
admissible regime

$0 < \beta \le 1$.

Deriving the sharper interval selected by lepton and quark hierarchies requires
additional geometric input beyond the present argument.

# Open Directions

1. **Sharper derivation of $\beta$**  
   From refined geometry of the relational front and the effective dimension of
   the explored configuration space

2. **Connection with the phenomenological window**  
   Explaining why the hierarchy-compatible regime appears near the small-$\beta$
   range identified in O3

3. **Extension to quarks and neutrinos**  
   Testing whether the same bounded-growth logic constrains the full flavour sector

4. **Beyond power-law ansätze**  
   Studying whether more general admissible growth laws obey analogous structural
   bounds

# Status

This framework is now:

- spectrally complete
- dynamically unified
- structurally constrained
- free of superlinear cascade arbitrariness

It does not assume:

- unconstrained hierarchy parameters
- arbitrary graph-growth laws
- external amplification mechanisms

# Repository Structure
```
paper/
├── out/ # Compiled O5 PDF
├── tex/ # LaTeX sources
└── README.md
```
# Citation

If you reference this work, please cite:

J. Beau, Subdiffusive Valence Growth under Bounded Relational Flux: Structural Derivation of the Cascade Exponent, Zenodo, 2026.

# Acknowledgements

Portions of the derivations, conceptual synthesis, and editorial refinement
benefited from iterative interactions with large language models used as
analytical assistants.
All theoretical results and interpretations remain the sole responsibility
of the author.

# Contributions
 
This repository is intended as a research reference.

Critical feedback, independent verification, and alternative derivations of the
cascade exponent bound are welcome.

Please open an issue to discuss conceptual points,
technical details, or possible extensions.
