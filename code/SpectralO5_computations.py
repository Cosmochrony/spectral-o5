#!/usr/bin/env python3
"""
SpectralO5 -- Numerical Computations
======================================
All scripts supporting the paper
  "Finite Character-Trace Saturation and the Limits of Vertex-Based Cascade Dynamics"
  J. Beau, Cosmochrony O-Series, 2026

Sections:
  §1  Shared utilities (finite-field arithmetic, quaternions)
  §2  Group/graph construction: PSL(2,F_q) or PGL(2,F_q) via the P^1(F_q)
      permutation representation (scalar-invariant canonicalization)
  §3  Character table via Dixon's algorithm, verified via Burnside's
      identity and commutativity of the class-sum matrices
  §4  Character-trace layer (M_tr, kappa_bar_rho: a TRACE AVERAGE
      tr(A_rho)/dim(rho), not an eigenvalue of A_rho = sum_s rho(s) --
      the generating set is only 6 elements of a much larger conjugacy
      class, so A_rho is not central in general; see CharacterTraceModel)
  §5  Shell-by-shell (graph-distance) cascade
  §6  Figure 1: Version A vs B (character-based) on X^{5,13}
  §7  Figure 2: Version B character, q-dependence on X^{5,q}
  §8  Figure 3: Matrix-based variants M1/M2/M3/M4 on X^{5,13}
  §9  Figure 4: Steinberg St_elem pre-saturation law, q in {13,17,29}
  §10 Main: generate all figures

Usage:
  python3 SpectralO5_computations.py          # generate all figures
  python3 SpectralO5_computations.py --fig N  # generate figure N only (1-4)

Dependencies: numpy, scipy, matplotlib, sympy (standard scientific stack)

Verification discipline: the group is verified (in `build_graph`/`CharacterTraceModel.__init__`)
against the closed-form order of PSL(2,F_q)/PGL(2,F_q), exact (p+1)-regularity, and full edge
symmetry; the character table is verified (in `dixon_character_table`) via exhaustive pairwise
commutativity of the class-sum matrices, Burnside's identity (sum of squared dimensions = |G|),
reality of the recovered values, and full row/column orthogonality of the table -- every check
raises AssertionError on failure rather than silently proceeding.
"""

import sys
import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import deque
from itertools import permutations, product as iproduct
from sympy.combinatorics import Permutation, PermutationGroup


# ================================================================
# §1  SHARED UTILITIES
# ================================================================

def mod_inv(a, q):
  """Modular inverse of a mod q (q prime)."""
  return pow(int(a), q - 2, q)


def sqrt_mod(a, q):
  """Square root of a mod q using Tonelli-Shanks. Returns None if not square."""
  a = int(a) % q
  if a == 0:
    return 0
  if pow(a, (q - 1) // 2, q) != 1:
    return None
  if q % 4 == 3:
    return pow(a, (q + 1) // 4, q)
  s, n = 0, q - 1
  while n % 2 == 0:
    s += 1
    n //= 2
  z = 2
  while pow(z, (q - 1) // 2, q) != q - 1:
    z += 1
  M, c, t, R = s, pow(z, n, q), pow(a, n, q), pow(a, (n + 1) // 2, q)
  while True:
    if t == 1:
      return R
    i, tmp = 1, (t * t) % q
    while tmp != 1:
      tmp = (tmp * tmp) % q
      i += 1
    b = pow(c, 1 << (M - i - 1), q)
    M, c, t, R = i, (b * b) % q, (t * b * b) % q, (R * b) % q


def is_square(a, q):
  """Check if a is a quadratic residue mod q."""
  return a == 0 or pow(int(a) % q, (q - 1) // 2, q) == 1


def mat_mul_mod(A, B, q):
  """Multiply two 2x2 integer matrices mod q."""
  return np.array([
    [(A[0, 0] * B[0, 0] + A[0, 1] * B[1, 0]) % q,
     (A[0, 0] * B[0, 1] + A[0, 1] * B[1, 1]) % q],
    [(A[1, 0] * B[0, 0] + A[1, 1] * B[1, 0]) % q,
     (A[1, 0] * B[0, 1] + A[1, 1] * B[1, 1]) % q]
  ], dtype=np.int64)


def four_square(p):
  """Find ONE (a,b,c,d) with a^2+b^2+c^2+d^2=p, a>0 odd, b,c,d even."""
  for a in range(1, p, 2):
    for b in range(0, p, 2):
      for c in range(0, p, 2):
        d2 = p - a * a - b * b - c * c
        if d2 < 0:
          continue
        d = int(d2 ** 0.5 + 0.5)
        if d * d == d2 and d % 2 == 0:
          return a, b, c, d


def find_lps_quaternions(p):
  """
  The exact set of p+1 quaternions (a,b,c,d) with a^2+b^2+c^2+d^2=p, a>0 ODD,
  b,c,d even (Jacobi's four-square count restricted to this sign/parity
  class is exactly p+1 for prime p = 1 mod 4). Requiring a odd (not just
  positive) is essential: without it, permuting the found quaternion's
  coordinates can put an EVEN entry in the first slot, inflating the
  enumerated set (12 elements for p=5 instead of 6) and making the
  downstream choice of generators depend on arbitrary set-iteration order.
  This set is automatically closed under (a,b,c,d) -> (a,-b,-c,-d)
  (quaternion conjugation), which is what makes the resulting Cayley graph
  generating set symmetric once canonicalized projectively (see
  `build_graph`).
  """
  a0, b0, c0, d0 = four_square(p)
  parts = [a0, b0, c0, d0]
  seen = set()
  for perm in permutations(range(4)):
    for signs in iproduct([-1, 1], repeat=4):
      vals = tuple(signs[j] * parts[perm[j]] for j in range(4))
      if vals[0] > 0 and vals[0] % 2 == 1 and sum(v * v for v in vals) == p:
        seen.add(vals)
  assert len(seen) == p + 1, f"expected {p + 1} LPS quaternions for p={p}, got {len(seen)}"
  return sorted(seen)


def quat_to_mat(quat, q, i_val):
  aa, bb, cc, dd = quat
  return np.array([[(aa + bb * i_val) % q, (cc + dd * i_val) % q],
                    [(-cc + dd * i_val) % q, (aa - bb * i_val) % q]], dtype=np.int64)


def action_on_P1(M, q):
  """Permutation of P^1(F_q) induced by M (any GL(2,F_q) representative).
  Returns array of length q+1. This is scalar-invariant: M and lambda*M
  induce the identical permutation, which is what makes it usable as a
  canonical key for elements of PGL(2,F_q)/PSL(2,F_q)."""
  a, b, c, d = int(M[0, 0]), int(M[0, 1]), int(M[1, 0]), int(M[1, 1])
  sigma = np.zeros(q + 1, dtype=np.int64)
  for i in range(q):
    num = (a * i + b) % q
    den = (c * i + d) % q
    sigma[i] = (num * mod_inv(den, q)) % q if den != 0 else q
  sigma[q] = (a * mod_inv(c, q)) % q if c != 0 else q
  return sigma


# ================================================================
# §2  GROUP / GRAPH CONSTRUCTION
# ================================================================
#
# v2.1 replaces the v1.1/v2.0 construction, which canonicalized a matrix by
# quotienting only by {+-I} (`mat_to_key`'s min-with-negation trick) instead
# of by the full scalar group F_q^*. That under-quotienting produced a set
# larger than PSL(2,F_q) (by a q-dependent, arithmetically incoherent
# factor), and because the true matrix inverse of a determinant-p generator
# is a SCALAR multiple (1/p times) of its quaternion conjugate rather than
# equal to it, the resulting "Cayley graph" was directed: no tested edge had
# a reverse edge among the six generators (verified: 0 of 300 sampled edges
# were symmetric).
#
# Here, a group element is canonicalized by its induced PERMUTATION of
# P^1(F_q) (`action_on_P1`), which is exactly invariant under M -> lambda*M
# for any scalar lambda != 0 -- i.e. it is a faithful representative of the
# element's class in PGL(2,F_q). BFS closure under (correctly-normalized)
# matrix multiplication, deduplicated by this permutation key, therefore
# produces exactly the subgroup of PGL(2,F_q) generated by the quaternion
# generators -- either PSL(2,F_q) (order q(q^2-1)/2) or all of PGL(2,F_q)
# (order q(q^2-1)), depending on whether p is a quadratic residue mod q.
# Both cases are verified against the closed-form order below.

def build_graph(q, p=5):
  """
  Build the LPS Cayley graph X^{p,q} on PGL(2,F_q) or PSL(2,F_q) (whichever
  is generated). Returns: adj (adjacency list, vertex index -> [(neighbour
  index, generator index)]), elems (list of representative 2x2 matrices,
  one per group element), gen_mats, gen_quats, group_label ('PSL' or 'PGL').
  """
  gen_quats = find_lps_quaternions(p)
  i_val = next(x for x in range(1, q) if (x * x + 1) % q == 0)
  gen_mats = [quat_to_mat(quat, q, i_val) for quat in gen_quats]

  I = np.array([[1, 0], [0, 1]], dtype=np.int64)
  elems = [I]
  key0 = tuple(action_on_P1(I, q).tolist())
  key_to_idx = {key0: 0}
  adj = {0: []}
  bq = deque([0])
  while bq:
    u_idx = bq.popleft()
    u_mat = elems[u_idx]
    for gi, G in enumerate(gen_mats):
      v_mat = mat_mul_mod(u_mat, G, q)
      key = tuple(action_on_P1(v_mat, q).tolist())
      if key not in key_to_idx:
        v_idx = len(elems)
        key_to_idx[key] = v_idx
        elems.append(v_mat)
        adj[v_idx] = []
        bq.append(v_idx)
      else:
        v_idx = key_to_idx[key]
      adj[u_idx].append((v_idx, gi))

  n = len(elems)
  psl_order = q * (q * q - 1) // 2
  pgl_order = q * (q * q - 1)
  if n == psl_order:
    label = 'PSL'
  elif n == pgl_order:
    label = 'PGL'
  else:
    raise AssertionError(f"q={q}, p={p}: generated group has order {n}, "
                          f"matching neither |PSL(2,q)|={psl_order} nor |PGL(2,q)|={pgl_order}")
  degs = set(len(v) for v in adj.values())
  assert degs == {p + 1}, f"q={q}: expected {p + 1}-regular graph, got degrees {degs}"
  return adj, elems, gen_mats, gen_quats, label


def verify_symmetry(adj):
  """Every directed edge (u,gi)->v must have a reverse edge v->u. Returns
  the count of edges failing this (0 for a correctly-built Cayley graph)."""
  bad = 0
  for u, lst in adj.items():
    for (v, gi) in lst:
      if not any(vv == u for (vv, gj) in adj[v]):
        bad += 1
  return bad


def distinct_count(values, decimals=6):
  """Number of distinct values after rounding (avoids floating-point noise
  splitting a single true value into several)."""
  return len(set(np.round(np.asarray(values, dtype=float), decimals).tolist()))


def graph_adjacency_distinct_eigenvalues(adj, n, decimals=6):
  """Dense diagonalisation of the graph's own adjacency matrix. Only
  tractable for the smaller tested graphs (q=13: n=2184). Returns the
  number of distinct eigenvalues, used to make concrete the gap between
  the true spectrum and the sector-trace averages mu_bar_rho (see
  CharacterTraceModel and the module comment above SS4)."""
  A = np.zeros((n, n))
  for u, lst in adj.items():
    for (v, gi) in lst:
      A[u, v] = 1.0
  evals = np.linalg.eigvalsh(A)
  return distinct_count(evals, decimals)


# ================================================================
# §3  CHARACTER TABLE VIA DIXON'S ALGORITHM
# ================================================================
#
# The v1.1/v2.0 character table was hand-assembled from Dickson's
# classification and contained two independent bugs: the `sigma_l` discrete
# series characters were identical for every l (the closure captured `ll`
# but its body never referenced it), and the conjugacy-class-by-trace
# routine assumed determinant-1 matrices, which the mis-quotiented group of
# §2 (v1.1/v2.0) did not supply.
#
# Rather than hand-implement Dickson's formulas for both PSL(2,q) and
# PGL(2,q) a second time, the table here is computed from first principles
# via Dixon's algorithm: the class-sum structure constants a_{ijk} (number
# of ways g_k = x*y with x in class i, y in class j) are computed exactly
# from the group's own multiplication (fast: O(|G|*r) group operations,
# r = number of classes), the resulting r class-sum matrices M_i are
# verified to commute pairwise, and a generic real linear combination is
# diagonalized once; the common eigenvectors recover the central characters
# omega_rho(i) = |C_i| chi_rho(g_i)/dim(rho), from which dim(rho) and the
# full character table follow by the standard normalisation. This is
# verified twice before use: pairwise commutativity of the M_i (a necessary
# condition), and Burnside's identity sum_rho dim(rho)^2 = |G| (a strong
# sufficient check that is essentially impossible to satisfy by a wrong
# computation).

def dixon_character_table(perm_group):
  """
  perm_group: a sympy PermutationGroup.
  Returns: classes (list of sympy orbits), sizes (list[int]), reps
  (list[Permutation], one per class), cls_of (dict: element key tuple ->
  class index), dims (real array, length r), chartable (real r x r array,
  chartable[rho, k] = chi_rho at class k), id_idx (index of the identity
  class).
  """
  classes = perm_group.conjugacy_classes()
  r = len(classes)
  elt_key = lambda perm: tuple(perm.array_form)

  cls_of = {}
  reps = []
  sizes = []
  class_elems = []
  for i, C in enumerate(classes):
    elems = list(C)
    class_elems.append(elems)
    sizes.append(len(elems))
    reps.append(elems[0])
    for e in elems:
      cls_of[elt_key(e)] = i
  id_idx = next(i for i in range(r) if sizes[i] == 1)

  inv_cache = {}
  def inv_of(perm):
    key = elt_key(perm)
    if key not in inv_cache:
      inv_cache[key] = perm ** -1
    return inv_cache[key]

  M = [np.zeros((r, r)) for _ in range(r)]
  for i in range(r):
    for x in class_elems[i]:
      xinv = inv_of(x)
      for k in range(r):
        y = xinv * reps[k]
        j = cls_of[elt_key(y)]
        M[i][j, k] += 1

  # Verification 1: the class-sum matrices must pairwise commute. Checked
  # exhaustively (all r^2 ordered pairs, cheap for the r <= 19 tested here),
  # not sampled.
  rng = np.random.default_rng(0)
  for i in range(r):
    for j in range(r):
      if not np.allclose(M[i] @ M[j], M[j] @ M[i], atol=1e-6):
        raise AssertionError(f"class-sum matrices M[{i}],M[{j}] do not commute: "
                              "structure constants are wrong")

  coeffs = rng.standard_normal(r)
  Mc = sum(c * Mi for c, Mi in zip(coeffs, M))
  evals, evecs = np.linalg.eig(Mc)
  if len(set(np.round(evals.real, 6) + 1j * np.round(evals.imag, 6))) < r:
    raise AssertionError("random combination did not separate all classes; retry with a new seed")

  Gorder = perm_group.order()
  dims = []
  chartable = np.zeros((r, r), dtype=complex)
  for col in range(r):
    v = evecs[:, col]
    idx0 = np.argmax(np.abs(v))
    omega = np.array([(Mi @ v)[idx0] / v[idx0] for Mi in M])
    denom = sum((omega[i] ** 2) / sizes[i] for i in range(r))
    dim = np.sqrt(Gorder / denom)
    dims.append(dim)
    chartable[col, :] = dim * omega / np.array(sizes)
  dims = np.array(dims)

  # Verification 2: Burnside's identity sum dim(rho)^2 = |G|.
  burnside = np.sum((dims.real) ** 2)
  if abs(burnside - Gorder) > 1e-3 * Gorder:
    raise AssertionError(f"Burnside check failed: sum(dim^2)={burnside}, |G|={Gorder}")
  # Verification 3: character values and dimensions must be real (holds
  # here because q = 1 mod 4 for every q tested by this paper).
  if np.max(np.abs(dims.imag)) > 1e-6 or np.max(np.abs(chartable.imag)) > 1e-6:
    raise AssertionError("character table has non-negligible imaginary part")
  chartable = chartable.real
  dims = dims.real

  # Verification 4: full row and column orthogonality of the character
  # table (not just Burnside's diagonal identity). Row rho, rho': sum_k
  # |C_k| chi_rho(k) chi_rho'(k) = |G| delta_{rho,rho'}. Column k, k':
  # sum_rho chi_rho(k) chi_rho(k') = (|G|/|C_k|) delta_{k,k'}.
  sizes_arr = np.array(sizes, dtype=float)
  row_gram = (chartable * sizes_arr[None, :]) @ chartable.T
  if not np.allclose(row_gram, Gorder * np.eye(r), atol=1e-4 * Gorder):
    raise AssertionError("row orthogonality of the character table failed")
  col_gram = chartable.T @ chartable
  expected_col = np.diag(Gorder / sizes_arr)
  if not np.allclose(col_gram, expected_col, atol=1e-4 * Gorder):
    raise AssertionError("column orthogonality of the character table failed")

  return classes, sizes, reps, cls_of, dims, chartable, id_idx


# ================================================================
# §4  CHARACTER-TRACE LAYER
# ================================================================
#
# mu_bar_rho = tr(A_rho)/dim(rho), A_rho = sum_{s in generators} rho(s), is a
# TRACE AVERAGE, not an eigenvalue of A_rho: the 6 generators are only a
# small part of their conjugacy class (size 182/306/870 for q=13/17/29), so
# A_rho is generally not central and has (up to dim(rho)) distinct genuine
# eigenvalues. For q=13, the actual graph adjacency spectrum has 55 distinct
# eigenvalues against the 15 sector-trace values computed here. Every
# "trace-selected"/kappa/M_tr quantity below is named and used accordingly:
# none of it is identified with a Laplacian eigenvalue, a spectral envelope,
# or a Ramanujan-admissible mode. See SpectralO5.tex, Remark
# rem:trace-not-eigenvalue, for the full statement and the class-function
# saturation result (Proposition prop:vertex-admissible-saturation) that
# holds regardless of this distinction.

class CharacterTraceModel:
  """Bundles the verified group/graph/character-table data for one (q,p)
  and exposes the per-vertex character-trace fingerprint pi_A(g). See the
  module-level comment above §4: kappa/mu_bar here are trace averages, not
  proven spectral eigenvalues."""

  def __init__(self, q, p=5):
    self.q, self.p, self.d = q, p, p + 1
    adj, elems, gen_mats, gen_quats, label = build_graph(q, p)
    bad = verify_symmetry(adj)
    if bad:
      raise AssertionError(f"q={q}: {bad} asymmetric edges in the Cayley graph")
    self.adj, self.elems, self.gen_mats, self.gen_quats, self.label = \
      adj, elems, gen_mats, gen_quats, label
    self.n = len(elems)

    gen_perms = [Permutation(action_on_P1(G, q).tolist()) for G in gen_mats]
    self.perm_group = PermutationGroup(gen_perms)
    assert self.perm_group.order() == self.n

    (self.classes, self.sizes, self.reps, self.cls_of,
     self.dims, self.chartable, self.id_idx) = dixon_character_table(self.perm_group)
    self.r = len(self.classes)

    gen_key = tuple(gen_perms[0].array_form)
    self.gen_class = self.cls_of[gen_key]
    self.gen_class_ids = sorted({self.cls_of[tuple(gp.array_form)] for gp in gen_perms})

    d = self.d
    mu = d * self.chartable[:, self.gen_class] / self.dims
    lam = d - mu
    lam_star = (np.sqrt(d - 1) + 1) ** 2
    self.lam, self.lam_star = lam, lam_star
    self.tr_mask = (lam > 1e-6) & (lam <= lam_star + 1e-9)
    self.n_tr = int(self.tr_mask.sum())
    self.kappa = mu[self.tr_mask] / d
    # M_tr: rows = conjugacy classes, columns = trace-selected irreps
    self.M_tr = self.chartable[self.tr_mask, :].T
    self.rank_Mtr = int(np.linalg.matrix_rank(self.M_tr, tol=1e-6))
    self.n_zero_kappa = int(np.sum(np.abs(self.kappa) < 1e-9))

    # P_A = M_tr @ diag(kappa): this, not M_tr itself, is what actually
    # determines dim(R_A). A row of M_tr is transformed component-wise by
    # kappa_rho before it ever reaches pi_A; kappa_rho = 0 is compatible with
    # trace-selection (lambda_rho = d exactly still lies in (0, lambda*]),
    # so a column of M_tr can be trace-selected yet contribute nothing to
    # the achievable span. r_A = rank(P_A) <= rank(M_tr), and the inequality is
    # observed to be strict in practice (see CharacterTraceModel.summary()).
    self.P_A = self.M_tr * self.kappa[None, :]
    self.r_A = int(np.linalg.matrix_rank(self.P_A, tol=1e-6))

    # Precompute chi_rho(g) and pi_A(g) for every group element (row = vertex
    # index). Chi_mat stores the BARE characters directly from the table
    # lookup, not recovered by dividing Pi_mat by kappa (kappa can be as
    # small as 1e-19 for some trace-selected sectors, which is numerically
    # fragile as a division and unnecessary when the bare values are already
    # available from chartable).
    elem_class = np.array([
      self.cls_of[tuple(action_on_P1(M, q).tolist())] for M in elems
    ])
    self.Chi_mat = self.chartable[self.tr_mask][:, elem_class].T
    # Chi_mat[v, :] = (chi_rho(v))_{rho in Gtr}
    self.Pi_mat = self.Chi_mat * self.kappa[None, :]
    # Pi_mat[v, :] = pi_A(v) in R^{n_tr}; rank(Pi_mat) == r_A by construction.

  def pi_A(self, v_idx):
    return self.Pi_mat[v_idx, :]

  def summary(self):
    return (f"q={self.q} ({self.label}(2,{self.q})): |G|={self.n}, r={self.r} classes, "
            f"generators in {len(self.gen_class_ids)} class(es), "
            f"n_tr={self.n_tr}/{self.r}, rank(M_tr)={self.rank_Mtr}, "
            f"r_A=rank(P_A)={self.r_A} ({self.n_zero_kappa} trace-selected sectors have kappa=0)")


# ================================================================
# §5  SHELL-BY-SHELL (GRAPH-DISTANCE) CASCADE
# ================================================================
#
# Unchanged from v2.0: shell m is tested against the span of strictly
# earlier shells, then absorbed as a whole. This removes all dependence on
# BFS tie-breaking, parent choice, or insertion order within a shell.

def bfs_shells(adj, n_verts):
  """Exact graph-distance shells and distances from vertex 0."""
  dist = {0: 0}
  bq = deque([0])
  while bq:
    u = bq.popleft()
    for (v, gi) in adj.get(u, []):
      if v not in dist:
        dist[v] = dist[u] + 1
        bq.append(v)
  shells = {}
  for v, d in dist.items():
    shells.setdefault(d, []).append(v)
  return shells, dist


def _rank_basis(sp):
  if len(sp) == 0:
    return 0, None
  sv = np.linalg.svd(sp, compute_uv=False)
  r = int(np.sum(sv > 1e-8))
  if r == 0:
    return 0, None
  _, _, Vt = np.linalg.svd(sp, full_matrices=False)
  return r, Vt[:r, :]


def _is_novel(fp, rank, Vt):
  if rank == 0 or np.all(np.abs(fp) < 1e-10):
    return True
  proj = Vt.T @ (Vt @ fp)
  return np.linalg.norm(fp - proj) > 1e-8


def layered_vertex_cascade(shells, dist, fp_func, fp_dim, max_shell=None):
  """
  Version A (vertex-based): shell m is tested against Pi_A(S_{m-1}), the
  span of strictly earlier shells, then absorbed. fp_func(v) -> R^fp_dim.
  Returns dict of arrays: Sn (cumulative |S_n|), shell, dim, novel, rt.
  """
  max_d = max(shells.keys()) if max_shell is None else max_shell
  span = np.zeros((0, fp_dim))
  res = {'Sn': [], 'shell': [], 'dim': [], 'novel': [], 'rt': []}
  cum = 0
  for m in range(0, max_d + 1):
    verts = shells.get(m, [])
    rank, Vt = _rank_basis(span)
    novel = 0
    rows = []
    for v in verts:
      f = fp_func(v)
      if _is_novel(f, rank, Vt):
        novel += 1
      rows.append(f)
    if rows:
      span = np.vstack([span, np.array(rows)]) if len(span) > 0 else np.array(rows)
    cum += len(verts)
    rank2, _ = _rank_basis(span)
    res['Sn'].append(cum)
    res['shell'].append(m)
    res['dim'].append(rank2)
    res['novel'].append(novel)
    res['rt'].append(novel / len(verts) if verts else 0.0)
  return {k: np.array(v) for k, v in res.items()}


def layered_transition_cascade(adj, shells, dist, fp_func, fp_dim, max_shell=None):
  """
  Version B (layered, non-tautological): the edges from shell m-1 into
  shell m are tested against Pi_B^{<m}, the span of transitions landing in
  strictly earlier shells, then absorbed as a whole. fp_func(u, v, gi) ->
  R^fp_dim. A boundary vertex v in shell m is counted as novel if *any*
  edge into it from shell m-1 has a fingerprint outside Pi_B^{<m}.
  """
  max_d = max(shells.keys()) if max_shell is None else max_shell
  span = np.zeros((0, fp_dim))
  res = {'Sn': [], 'shell': [], 'dim': [], 'novel': [], 'rt': []}
  cum = 0
  for m in range(0, max_d + 1):
    verts = shells.get(m, [])
    rank, Vt = _rank_basis(span)
    novel_verts = set()
    rows = []
    if m >= 1:
      for u in shells.get(m - 1, []):
        for (v, gi) in adj.get(u, []):
          if dist.get(v) == m:
            f = fp_func(u, v, gi)
            if _is_novel(f, rank, Vt):
              novel_verts.add(v)
            rows.append(f)
    if rows:
      span = np.vstack([span, np.array(rows)]) if len(span) > 0 else np.array(rows)
    cum += len(verts)
    rank2, _ = _rank_basis(span)
    res['Sn'].append(cum)
    res['shell'].append(m)
    res['dim'].append(rank2)
    res['novel'].append(len(novel_verts))
    res['rt'].append(len(novel_verts) / len(verts) if verts else 0.0)
  return {k: np.array(v) for k, v in res.items()}


def build_steinberg_projections(elems, gen_mats, q):
  """Steinberg projections tilde_sigma for all vertices and generators."""
  steins = np.zeros((len(elems), q + 1))
  for i, M in enumerate(elems):
    s = action_on_P1(M, q).astype(float)
    steins[i] = s - s.mean()
  gen_steins = np.zeros((len(gen_mats), q + 1))
  for gi, G in enumerate(gen_mats):
    s = action_on_P1(G, q).astype(float)
    gen_steins[gi] = s - s.mean()
  return steins, gen_steins


# ================================================================
# §6  FIGURE 1 -- Version A vs B (character-based) on X^{5,13}
# ================================================================

def figure1_A_vs_B(q=13, p=5, outfile='fig1_A_vs_B.png'):
  print(f"Figure 1: Version A vs B (character-based), q={q}")
  model = CharacterTraceModel(q, p)
  print("  " + model.summary())

  if q == 13:
    # Live check of the trace-average/eigenvalue gap claimed throughout
    # the paper (Remark rem:trace-not-eigenvalue): computed here, not just
    # asserted in prose.
    n_eig = graph_adjacency_distinct_eigenvalues(model.adj, model.n)
    lam_all = model.d - model.d * model.chartable[:, model.gen_class] / model.dims
    n_trace = distinct_count(lam_all)
    print(f"  eigenvalue-gap check: {n_eig} distinct graph eigenvalues vs "
          f"{n_trace} distinct trace averages across {model.r} sectors")
    assert n_eig == 55 and n_trace == 9, (
      f"expected 55 distinct eigenvalues / 9 distinct trace averages for q=13, "
      f"got {n_eig} / {n_trace}")

  shells, dist = bfs_shells(model.adj, model.n)

  def fp_A(v):
    return model.pi_A(v)

  # pi_B(u->v) = kappa * chi(u) * chi(v); Chi_mat already holds the bare
  # (un-scaled) characters chi_rho(v) directly from the table.
  bare_chi = model.Chi_mat

  def fp_B(u, v, gi):
    return model.kappa * bare_chi[u, :] * bare_chi[v, :]

  res_A = layered_vertex_cascade(shells, dist, fp_A, model.n_tr)
  res_B = layered_transition_cascade(model.adj, shells, dist, fp_B, model.n_tr)

  fig, axes = plt.subplots(2, 2, figsize=(12, 8))
  fig.suptitle(
    f'Version A vs B (character-based) on $X^{{{p},{q}}}$, shell-layered cascade\n'
    f'{model.label}(2,{q}): $|G|={model.n}$, $n_{{\\rm tr}}={model.n_tr}$, '
    f'$\\mathrm{{rank}}(M_{{\\rm tr}})={model.rank_Mtr}$, $r_A=\\mathrm{{rank}}(P_A)={model.r_A}$',
    fontsize=12
  )
  Sn_A, Sn_B = res_A['Sn'], res_B['Sn']

  ax = axes[0, 0]
  ax.plot(Sn_A, res_A['dim'], 'g-o', ms=3, label=r'$\dim\Pi_A(S_n)$')
  ax.plot(Sn_B, res_B['dim'], 'b-s', ms=3, label=r'$\dim\Pi_B^{<n}$', alpha=0.8)
  ax.axhline(model.r_A, color='g', linestyle='--',
             label=f'$r_A=\\mathrm{{rank}}(P_A)={model.r_A}$')
  ax.axhline(model.rank_Mtr, color='darkgreen', linestyle='-.', alpha=0.6,
             label=f'$\\mathrm{{rank}}(M_{{\\rm tr}})={model.rank_Mtr}$ (unreachable bound)')
  ax.axhline(model.n_tr, color='b', linestyle=':',
             label=f'$n_{{\\rm tr}}={model.n_tr}$')
  ax.set_xlabel(r'$|S_n|$ (log)')
  ax.set_xscale('log')
  ax.set_ylabel(r'$\dim\Pi(S_n)$')
  ax.set_title('Rank of character-trace span (shell-layered)')
  ax.legend(fontsize=7)
  ax.grid(True, alpha=0.3)

  ax = axes[0, 1]
  ax.plot(Sn_A, res_A['rt'], 'g-o', ms=3, label=r'$\tilde{r}^A_n$ (vertex)')
  ax.plot(Sn_B, res_B['rt'], 'b-s', ms=3, label=r'$\tilde{r}^B_n$ (character, layered)', alpha=0.8)
  ax.set_xlabel(r'$|S_n|$ (log)')
  ax.set_xscale('log')
  ax.set_ylabel(r'$\tilde{r}_n$')
  ax.set_title(r'Shell-wise novelty fraction')
  ax.legend(fontsize=9)
  ax.grid(True, alpha=0.3)
  ax.set_ylim(-0.05, 1.05)

  ax = axes[1, 0]
  rn_ratio = np.array(res_A['dim'], dtype=float) / np.maximum(res_A['Sn'], 1)
  ax.plot(Sn_A, rn_ratio, 'g-o', ms=3, label=r'$r_n$')
  ax.axhline(model.r_A / model.n, color='k', linestyle=':', alpha=0.6,
             label=r'$r_A/|G|$ (finite floor, not 0)')
  ax.set_xlabel(r'$|S_n|$ (log)')
  ax.set_xscale('log')
  ax.set_ylabel(r'$r_n = \dim\Pi_A(S_n)/|S_n|$')
  ax.set_title(r'Finite bound $r_n\leq r_A/|S_n|$ (Corollary~1)')
  ax.legend(fontsize=8)
  ax.grid(True, alpha=0.3)

  ax = axes[1, 1]
  ax.axis('off')
  satA = next((res_A['Sn'][i] for i, d_ in enumerate(res_A['dim']) if d_ >= model.r_A), None)
  summary = (
    f"KEY NUMBERS ($q={q}$, $p={p}$), shell-layered:\n\n"
    f"  Group: {model.label}(2,{q})\n"
    f"  $|G| = {model.n}$\n"
    f"  Generators in {len(model.gen_class_ids)} conjugacy class(es)\n"
    f"  $n_{{\\rm tr}} = {model.n_tr}$ of ${model.r}$ classes\n"
    f"  $\\mathrm{{rank}}(M_{{\\rm tr}}) = {model.rank_Mtr}$\n"
    f"  $r_A = \\mathrm{{rank}}(P_A) = {model.r_A}$"
    f" ({model.n_zero_kappa} trace-selected sectors have $\\kappa_\\rho=0$)\n\n"
    f"VERSION A (empirical, this traversal only):\n"
    f"  $\\dim\\Pi_A(S_n)$ reaches $r_A$ at $|S_n| \\approx {satA}$\n\n"
    f"VERSION B (character, layered):\n"
    f"  Max $\\dim\\Pi_B^{{<n}}$: {max(res_B['dim'])}\n"
    f"  Late $\\tilde{{r}}^B$: {np.mean(res_B['rt'][-5:]):.3f}"
  )
  ax.text(0.05, 0.95, summary, transform=ax.transAxes,
          fontsize=9, verticalalignment='top', fontfamily='monospace',
          bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
  ax.set_title('Summary')

  plt.tight_layout()
  plt.savefig(outfile, dpi=150, bbox_inches='tight')
  print(f"  Saved {outfile}")


# ================================================================
# §7  FIGURE 2 -- Version B character q-dependence
# ================================================================

def figure2_versionB_qdep(qs=(13, 29), p=5, outfile='fig2_versionB_sat.png'):
  print(f"Figure 2: Version B character q-dependence, q={qs}")

  fig, axes = plt.subplots(1, 2, figsize=(12, 5))
  fig.suptitle(
    r'Character-based Version B: $q$-dependence on $X^{5,q}$',
    fontsize=12
  )
  colors = {13: 'green', 17: 'blue', 29: 'orange', 41: 'red'}

  for q in qs:
    model = CharacterTraceModel(q, p)
    print("  " + model.summary())
    shells, dist = bfs_shells(model.adj, model.n)
    bare_chi = model.Chi_mat

    def fp_A(v, Pi=model.Pi_mat):
      return Pi[v, :]

    def fp_B(u, v, gi, kap=model.kappa, chi=bare_chi):
      return kap * chi[u, :] * chi[v, :]

    res_A = layered_vertex_cascade(shells, dist, fp_A, model.n_tr)
    res_B = layered_transition_cascade(model.adj, shells, dist, fp_B, model.n_tr)
    col = colors.get(q, 'gray')

    axes[0].plot(res_A['Sn'], res_A['dim'],
                 color=col, linestyle='-', marker='o', ms=2,
                 label=f'$\\dim\\Pi_A$, $q={q}$')
    axes[0].plot(res_B['Sn'], res_B['dim'],
                 color=col, linestyle='--', marker='s', ms=2,
                 label=f'$\\dim\\Pi_B^{{<n}}$, $q={q}$', alpha=0.7)
    axes[0].axhline(model.r_A, color=col, linestyle=':', alpha=0.4)

    axes[1].plot(res_A['Sn'], res_A['rt'],
                 color=col, linestyle='-', marker='o', ms=2,
                 label=f'$\\tilde{{r}}^A$, $q={q}$')
    axes[1].plot(res_B['Sn'], res_B['rt'],
                 color=col, linestyle='--', marker='s', ms=2,
                 label=f'$\\tilde{{r}}^B$, $q={q}$', alpha=0.7)

  axes[0].set_xlabel(r'$|S_n|$ (log)')
  axes[0].set_xscale('log')
  axes[0].set_ylabel(r'$\dim\Pi(S_n)$')
  axes[0].set_title('Rank of character-trace spans')
  axes[0].legend(fontsize=7)
  axes[0].grid(True, alpha=0.3)

  axes[1].set_xlabel(r'$|S_n|$ (log)')
  axes[1].set_xscale('log')
  axes[1].set_ylabel(r'$\tilde{r}_n$')
  axes[1].set_title(r'Trace-productive frontier fraction')
  axes[1].legend(fontsize=7)
  axes[1].grid(True, alpha=0.3)
  axes[1].set_ylim(-0.05, 1.05)

  plt.tight_layout()
  plt.savefig(outfile, dpi=150, bbox_inches='tight')
  print(f"  Saved {outfile}")


# ================================================================
# §8  FIGURE 3 -- Matrix variants M1/M2/M3/M4 on X^{5,13}
# ================================================================

def figure3_matrix_variants(q=13, p=5, outfile='fig3_matB_variants.png'):
  print(f"Figure 3: Matrix variants M1/M2/M3/M4, q={q}")
  model = CharacterTraceModel(q, p)
  print("  " + model.summary())
  bare_chi = model.Chi_mat
  shells, dist = bfs_shells(model.adj, model.n)

  def fp_M1(u, v, gi):
    return model.elems[v].astype(float).flatten() / q

  def fp_M2(u, v, gi):
    Mu = model.elems[u].astype(float) / q
    Ms = model.gen_mats[gi].astype(float) / q
    return np.outer(Mu.flatten(), Ms.flatten()).flatten()

  def fp_M3(u, v, gi):
    chi_u = bare_chi[u, :]
    Ms = model.gen_mats[gi].astype(float) / q
    return np.outer(model.kappa * chi_u, Ms.flatten()).flatten()

  def fp_M4(u, v, gi):
    Mu = model.elems[u].astype(float) / q
    Ms = model.gen_mats[gi].astype(float) / q
    return np.concatenate([(Mu @ Ms).flatten(),
                           (Mu.T @ Ms).flatten(),
                           (Mu @ Ms.T).flatten()])

  variants = [
    ('M1: $\\mathrm{vec}(M_v)$', fp_M1, 4, 'gray'),
    ('M2: $M_u\\otimes M_s$', fp_M2, 16, 'blue'),
    ('M3: $\\chi_\\rho(u)\\otimes\\mathrm{vec}(M_s)$', fp_M3, model.n_tr * 4, 'orange'),
    ('M4: 3 products', fp_M4, 12, 'green'),
  ]

  fig, axes = plt.subplots(1, 3, figsize=(15, 5))
  fig.suptitle(
    f'Matrix-based Version B variants on $X^{{{p},{q}}}$, shell-layered cascade',
    fontsize=12
  )

  for name, fp, dim, col in variants:
    print(f"  Running {name} (dim={dim})...", end=' ', flush=True)
    res = layered_transition_cascade(model.adj, shells, dist, fp, dim)
    print(f"max_dim={max(res['dim'])}, late_rt={np.mean(res['rt'][-5:]):.3f}")
    Sn = res['Sn']
    axes[0].plot(Sn, res['dim'], marker='o', ms=2, color=col, label=name)
    axes[0].axhline(dim, color=col, linestyle=':', alpha=0.3)
    axes[1].plot(Sn, res['rt'], marker='o', ms=2, color=col, label=name)
    valid = np.array(res['rt']) > 0.005
    Sn_arr = np.array(Sn)
    rt_arr = np.array(res['rt'])
    if valid.sum() > 3:
      axes[2].loglog(Sn_arr[valid], rt_arr[valid],
                     marker='o', ms=2, color=col, label=name)

  axes[0].set_xlabel(r'$|S_n|$ (log)')
  axes[0].set_xscale('log')
  axes[0].set_ylabel(r'$\dim\Pi^{\rm mat}_{<n}(S_n)$')
  axes[0].set_title('Rank of matrix span (shell-layered)')
  axes[0].legend(fontsize=7)
  axes[0].grid(True, alpha=0.3)

  axes[1].set_xlabel(r'$|S_n|$ (log)')
  axes[1].set_xscale('log')
  axes[1].set_ylabel(r'$\tilde{r}_n^{\rm mat}$')
  axes[1].set_title('Trace-productive frontier fraction')
  axes[1].legend(fontsize=7)
  axes[1].grid(True, alpha=0.3)
  axes[1].set_ylim(-0.05, 1.05)

  axes[2].set_xlabel(r'$|S_n|$ (log)')
  axes[2].set_ylabel(r'$\tilde{r}_n$ (log)')
  axes[2].set_title('Decay shape (log-log)')
  axes[2].legend(fontsize=7)
  axes[2].grid(True, alpha=0.3)

  plt.tight_layout()
  plt.savefig(outfile, dpi=150, bbox_inches='tight')
  print(f"  Saved {outfile}")


# ================================================================
# §9  FIGURE 4 -- Steinberg St_elem pre-saturation law
# ================================================================

def figure4_steinberg_presat(qs=(13, 17, 29), p=5, outfile='fig4_StElem_presat.png'):
  print(f"Figure 4: Steinberg St_elem pre-saturation, q={qs}")
  colors = {13: 'green', 17: 'blue', 29: 'orange'}

  all_data = {}
  for q in qs:
    print(f"  q={q}...", end=' ', flush=True)
    model = CharacterTraceModel(q, p)
    steins, gen_steins = build_steinberg_projections(model.elems, model.gen_mats, q)

    def fp_St(u, v, gi, st=steins, gst=gen_steins):
      return st[u] * gst[gi]

    shells, dist = bfs_shells(model.adj, model.n)
    res = layered_transition_cascade(model.adj, shells, dist, fp_St, q + 1,
                                      max_shell=min(len(shells) - 1, 400))

    Sn_arr = res['Sn']
    rt_arr = res['rt']
    dim_arr = res['dim']
    pp_arr = np.cumsum(res['novel'])

    sat_idx = next((i for i, d_ in enumerate(dim_arr) if d_ >= q + 1), None)
    sat_Sn = int(Sn_arr[sat_idx]) if sat_idx is not None else None

    hi = sat_idx if sat_idx is not None else len(Sn_arr)
    mask = (Sn_arr[:hi] > 3) & (pp_arr[:hi] > 0)
    beta_prod = None
    if mask.sum() >= 5:
      logx = np.log(Sn_arr[:hi][mask].astype(float))
      logy = np.log(pp_arr[:hi][mask].astype(float))
      slope, _ = np.polyfit(logx, logy, 1)
      beta_prod = slope

    all_data[q] = {
      'n': model.n, 'Sn': Sn_arr, 'rt': rt_arr, 'dim': dim_arr,
      'pp': pp_arr, 'sat_Sn': sat_Sn, 'sat_idx': sat_idx,
      'beta_prod': beta_prod, 'label': model.label
    }
    print(f"|S*|={sat_Sn}, beta_prod={beta_prod:.3f}" if beta_prod else f"|S*|={sat_Sn}")

  fig, axes = plt.subplots(2, 2, figsize=(12, 9))
  fig.suptitle(
    r'Steinberg $\tilde\sigma_u\odot\tilde\sigma_s$ pre-saturation law'
    f' -- $X^{{5,q}}$, $q\\in{{{",".join(map(str, qs))}}}$, shell-layered',
    fontsize=12
  )

  ax = axes[0, 0]
  for q, d in all_data.items():
    ax.plot(d['Sn'], d['rt'], color=colors[q], marker='o', ms=2,
            label=f'$q={q}$ ({d["label"]})')
    if d['sat_Sn']:
      ax.axvline(d['sat_Sn'], color=colors[q], linestyle='--', alpha=0.5)
  ax.set_xlabel(r'$|S_n|$ (log)')
  ax.set_xscale('log')
  ax.set_ylabel(r'$\tilde{r}_n^{\rm St}$')
  ax.set_title(r'Trace-productive frontier fraction')
  ax.legend(fontsize=9)
  ax.grid(True, alpha=0.3)
  ax.set_ylim(-0.05, 1.05)

  ax = axes[0, 1]
  for q, d in all_data.items():
    valid = d['rt'] > 0.005
    ax.loglog(d['Sn'][valid], d['rt'][valid], color=colors[q],
              marker='o', ms=2, label=f'$q={q}$')
  ax.set_xlabel(r'$|S_n|$ (log)')
  ax.set_ylabel(r'$\tilde{r}_n^{\rm St}$ (log)')
  ax.set_title('Decay shape (log-log)')
  ax.legend(fontsize=9)
  ax.grid(True, alpha=0.3)

  ax = axes[1, 0]
  for q, d in all_data.items():
    valid = d['pp'] > 0
    ax.loglog(d['Sn'][valid], d['pp'][valid], color=colors[q],
              marker='o', ms=2, label=f'$q={q}$')
  ax.set_xlabel(r'$|S_n|$ (log)')
  ax.set_ylabel(r'$p_n^{\rm prod}$ (log)')
  ax.set_title(r'Cumulative trace-productive front (reference only; no exponent fit)')
  ax.legend(fontsize=7)
  ax.grid(True, alpha=0.3)

  ax = axes[1, 1]
  ax.axis('off')
  lines = ['PARAMETER TABLE (shell-layered, verified group)\n']
  for q_v, d in all_data.items():
    lines.append(f'  $q={q_v}$ ({d["label"]}): $|G|={d["n"]}$, '
                 f'$|S^*|={d["sat_Sn"]}$, $|S^*|/|G|={d["sat_Sn"]/d["n"]:.5f}$')
  n_fit = sum(1 for d in all_data.values() if d['beta_prod'] is not None)
  lines.append('')
  lines.append(f'Usable pre-saturation windows for a beta_prod fit: {n_fit}/{len(all_data)}')
  lines.append('Saturation occurs inside the local tree-like regime')
  lines.append('shared by every tested q (shell sizes 1,6,30,150,... match')
  lines.append('the free 6-regular tree exactly before any collision).')

  ax.text(0.03, 0.97, '\n'.join(lines), transform=ax.transAxes,
          fontsize=9, verticalalignment='top', fontfamily='monospace',
          bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
  ax.set_title('Summary')

  plt.tight_layout()
  plt.savefig(outfile, dpi=150, bbox_inches='tight')
  print(f"  Saved {outfile}")


# ================================================================
# §10  MAIN
# ================================================================

def main():
  import argparse
  parser = argparse.ArgumentParser(description='SpectralO5 computations')
  parser.add_argument('--fig', type=int, default=0,
                      help='Generate only figure N (1-4). 0 = all.')
  parser.add_argument('--outdir', type=str, default='.',
                      help='Output directory for figures.')
  args = parser.parse_args()

  import os
  os.makedirs(args.outdir, exist_ok=True)

  def path(name):
    return os.path.join(args.outdir, name)

  figs = {
    1: lambda: figure1_A_vs_B(q=13, p=5,
                              outfile=path('fig1_A_vs_B.pdf')),
    2: lambda: figure2_versionB_qdep(qs=(13, 29), p=5,
                                     outfile=path('fig2_versionB_sat.pdf')),
    3: lambda: figure3_matrix_variants(q=13, p=5,
                                       outfile=path('fig3_matB_variants.pdf')),
    4: lambda: figure4_steinberg_presat(qs=(13, 17, 29), p=5,
                                        outfile=path('fig4_StElem_presat.pdf')),
  }

  if args.fig == 0:
    for i in sorted(figs):
      figs[i]()
  elif args.fig in figs:
    figs[args.fig]()
  else:
    print(f"Unknown figure {args.fig}. Choose 1-4.")


if __name__ == '__main__':
  main()
