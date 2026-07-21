r"""
Finite-domain valley-Hall waveguides: rod-conforming meshes and the scalar
time-harmonic solver.

Consolidates the mesh / phase-assignment / assembly helpers that were
copy-pasted across ``examples/valley_waveguide_tri.py``,
``examples/valley_robust.py`` and ``examples/valley_wg_gmsh.py``.  The copies had
drifted.  Most consequentially, ``valley_waveguide_tri.assemble`` accepted a
structural-damping argument ``eta`` and never applied it,

    return K - omega**2*M + 1j*omega*sp.diags(sponge*Md), M      # eta dropped

while ``valley_robust.assemble`` did apply it.  The frequency sweep that located
the guided band was therefore undamped and the robustness study that probed that
band was damped at eta=0.003: the two runs were never comparable.  With a single
implementation the divergence cannot recur.

Why the mesh conforms to the scatterers
---------------------------------------
The previous construction meshed a plain rectangle and then assigned each element
to the rod or matrix phase by testing whether its *centroid* fell inside a rod.
That is the staircased-boundary construction this project rejects everywhere else
-- see ``wavetopo/dolfinx_mesh.py`` for the through-hole argument, and the
conforming-void cloak, which reports markedly better cloaking than the same
problem posed with a SIMP void on a structured grid precisely because a
staircased rim scatters on its own account.

For a valley crystal the error is not merely quantitative.  The Dirac mass is
``m ~ cos(3*alpha)``: it *is* the C3-symmetry breaking of the scatterer.  A
staircase that perturbs the scatterer's threefold symmetry by an amount
comparable to the intended mass therefore perturbs the gap itself, and the
topological phase with it.

Earlier docstrings in this repo asserted that an unstructured Delaunay mesh
"respects C3 on average" and was therefore adequate.  That claim is false and is
not repeated here.  An unstructured mesh removes the *systematic* C3 bias of a
sheared structured grid, but centroid painting replaces it with a *random*,
realization-dependent bias with no controlled convergence.  Measured on three
painted realizations of nominally the same crystal, transmission to the far end
of the wall at omega = 7.267 was

    78 492 tris   T = 7.1e-03        (results/wg_tri_sweep.log)
    91 614 tris   T = 3.9e+01        (results/wg_tri_sweep2.log)
    97 520 tris   T = 4.2e-06        (results/valley_robust.log)

-- seven orders of magnitude at one frequency, spanning "strongly guided" to
"deep-gap floor".  No conclusion about topological protection survives that
spread.

Cutting the scatterers into the geometry makes the phase assignment exact: every
element lies wholly in one phase, each rod is exactly the intended equilateral
triangle at exactly the intended angle, and resolution is the only remaining mesh
parameter.  ``painted_mesh`` is retained *solely* so that the error of the old
construction can be measured rather than assumed.
"""
from __future__ import annotations

from collections import namedtuple

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

__all__ = [
    "Shape", "tri_rod_layout", "honeycomb_disk_layout",
    "conforming_mesh", "painted_mesh",
    "quadratic_sponge", "wall_polyline", "wall_side_fn",
    "assemble", "solve_transmission",
]

# Triangular-lattice primitive vectors (a1 = a*(1,0) is implicit).
A1 = np.array([1.0, 0.0])
A2 = np.array([0.5, np.sqrt(3.0) / 2.0])

#: A scatterer to be cut into the mesh.  ``kind`` is "poly" (``data`` is an
#: (n,2) vertex array, counter-clockwise) or "disk" (``data`` is ``(cx,cy,r)``).
Shape = namedtuple("Shape", "kind data")


# --------------------------------------------------------------------------- #
#  Lattice layouts
# --------------------------------------------------------------------------- #
def _lattice_sites(Lx, Ly, a=1.0, pad=2):
    """Rod-centre sites (cell centroids) covering [0,Lx]x[0,Ly] with a margin."""
    a1, a2 = a * A1, a * A2
    sites = [i * a1 + j * a2 + (a1 + a2) / 2
             for i in range(-pad, int(Lx / a) + pad + 1)
             for j in range(-pad, int(Ly / (a * A2[1])) + pad + 1)]
    return np.array(sites)


def _nearest_site(sites, xy):
    return int(np.argmin(np.hypot(sites[:, 0] - xy[0], sites[:, 1] - xy[1])))


def equilateral(center, alpha_deg, R):
    """Vertices of an equilateral triangle of circumradius ``R`` rotated by
    ``alpha_deg``.  Vertex convention matches the original ``paint_rods``:
    angles are ``alpha + pi/2 + 2*pi*k/3``, so alpha=30deg aligns the triangle's
    mirror planes with the lattice's (the C3v Dirac point, m=0)."""
    al = np.deg2rad(alpha_deg) + np.pi / 2 + 2 * np.pi * np.arange(3) / 3
    return np.asarray(center) + R * np.stack([np.cos(al), np.sin(al)], axis=1)


def tri_rod_layout(Lx, Ly, side_fn, *, a=1.0, R=0.44,
                   alpha_in=15.0, alpha_out=45.0,
                   skip_sites=(), angle_override=None, margin=None):
    """Mirror-partner triangular-rod valley crystal.

    ``side_fn(xy) -> bool`` selects the ``alpha_in`` domain; the complement gets
    ``alpha_out``.  The two angles should be mirror partners about the C3v point
    (``alpha`` and ``60 - alpha``) so the Dirac masses are equal and opposite and
    the interface carries |dC_v| = 1.

    ``skip_sites``   : iterable of (x,y); the nearest rod to each is omitted
                       (a vacancy defect).
    ``angle_override``: {(x,y): alpha_deg}; overrides the nearest rod's angle
                       (orientation disorder).

    Returns ``(shapes, centers, alphas)``.
    """
    sites = _lattice_sites(Lx, Ly, a=a)
    skip = {_nearest_site(sites, xy) for xy in skip_sites}
    over = {_nearest_site(sites, xy): float(al)
            for xy, al in (angle_override or {}).items()}

    # Drop rods that cannot reach the domain, so the boolean fragment stays cheap.
    m = R if margin is None else margin
    keep = ((sites[:, 0] > -m) & (sites[:, 0] < Lx + m) &
            (sites[:, 1] > -m) & (sites[:, 1] < Ly + m))

    shapes, centers, alphas = [], [], []
    for i in np.flatnonzero(keep):
        if i in skip:
            continue
        c = sites[i]
        al = over.get(i, alpha_in if side_fn(c) else alpha_out)
        shapes.append(Shape("poly", equilateral(c, al, R)))
        centers.append(c)
        alphas.append(al)
    return shapes, np.array(centers), np.array(alphas)


def honeycomb_disk_layout(Lx, Ly, side_fn, *, b=1.0, rA=0.46, rB=0.24,
                          margin=None):
    """Honeycomb of disks with a sublattice-radius imbalance -- the *other*
    valley mechanism, used by ``examples/valley_wg_gmsh.py``.

    Here the Dirac mass comes from the A/B radius difference rather than from a
    rotation angle, and it flips by swapping the two radii across the wall.  The
    mesh and solver below are identical; only the shape set differs.

    Lattice vectors and sublattice offsets follow ``valley_wg_gmsh.honeycomb_z``
    exactly: ``a1 = b(1.5, sqrt3/2)``, ``a2 = b(1.5, -sqrt3/2)``, A and B sites
    at ``(a1+a2)/3`` and ``2(a1+a2)/3``.  Note this is the honeycomb convention,
    *not* the triangular lattice of :func:`tri_rod_layout`.
    """
    s3 = np.sqrt(3.0)
    a1 = b * np.array([1.5, s3 / 2.0])
    a2 = b * np.array([1.5, -s3 / 2.0])
    subs = [(a1 + a2) / 3.0, 2.0 * (a1 + a2) / 3.0]      # A, B
    m = max(rA, rB) if margin is None else margin

    P = range(-1, int(Lx / (1.5 * b)) + 2)
    Q = range(-int(Ly / (0.9 * b)) - 2, int(Ly / (0.9 * b)) + 2)

    shapes, centers, radii = [], [], []
    for p in P:
        for q in Q:
            for s, off in enumerate(subs):
                c = off + p * a1 + q * a2
                if not (-m < c[0] < Lx + m and -m < c[1] < Ly + m):
                    continue
                # On the side_fn side, A carries rA and B carries rB; the radii
                # swap across the wall, which flips the sign of the Dirac mass.
                if side_fn(c):
                    r = rA if s == 0 else rB
                else:
                    r = rB if s == 0 else rA
                shapes.append(Shape("disk", (c[0], c[1], r)))
                centers.append(c)
                radii.append(r)
    return shapes, np.array(centers), np.array(radii)


# --------------------------------------------------------------------------- #
#  Meshing
# --------------------------------------------------------------------------- #
def _gmsh_session():
    import gmsh
    if gmsh.isInitialized():
        gmsh.finalize()
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    return gmsh


def _read_nodes(gmsh):
    tags, coord, _ = gmsh.model.mesh.getNodes()
    xy = coord.reshape(-1, 3)[:, :2]
    return xy, {int(t): i for i, t in enumerate(tags)}


def _prune_nodes(npx, tris):
    """Drop nodes referenced by no triangle and reindex.

    A scatterer overhanging the domain leaves a sliver surface outside the
    rectangle, which ``generate(2)`` still meshes.  Those nodes are attached to
    no kept element, so they would contribute empty rows to K and M and make D
    singular.  Removing the OCC surfaces instead is not safe -- a recursive
    delete would take boundary curves shared with the kept region.
    """
    used = np.zeros(npx.shape[0], bool)
    used[tris.ravel()] = True
    if used.all():
        return npx, tris
    remap = np.full(npx.shape[0], -1, np.int64)
    remap[used] = np.arange(int(used.sum()))
    return npx[used], remap[tris]


def conforming_mesh(Lx, Ly, h, shapes, *, h_scat=None, refine_dist=1.5):
    """Triangulate [0,Lx]x[0,Ly] with the mesh *conforming* to every scatterer.

    Each shape is cut into the rectangle with an OCC boolean fragment, so element
    edges follow scatterer boundaries exactly and the phase field ``z`` is
    exactly 0 or 1 per element -- no centroid test, no staircase.  Shapes may
    overhang the domain; the part outside is discarded by the fragment.

    ``h_scat`` (default ``h/2``) is the element size on scatterer boundaries,
    grading back to ``h`` over ``refine_dist * h``.

    Returns ``(npx, tris, z)`` with ``z=1`` inside the scatterers.
    """
    gmsh = _gmsh_session()
    gmsh.model.add("valley")
    occ = gmsh.model.occ

    rect = occ.addRectangle(0, 0, 0, Lx, Ly)
    tools = []
    for sh in shapes:
        if sh.kind == "disk":
            cx, cy, r = sh.data
            tools.append((2, occ.addDisk(cx, cy, 0, r, r)))
        elif sh.kind == "poly":
            pts = [occ.addPoint(x, y, 0) for x, y in sh.data]
            lns = [occ.addLine(pts[k], pts[(k + 1) % len(pts)])
                   for k in range(len(pts))]
            tools.append((2, occ.addPlaneSurface([occ.addCurveLoop(lns)])))
        else:                                            # pragma: no cover
            raise ValueError(f"unknown shape kind {sh.kind!r}")

    _, omap = occ.fragment([(2, rect)], tools)
    occ.synchronize()

    # omap[0] are the pieces of the rectangle; omap[1:] the pieces of each
    # scatterer.  A scatterer piece that is also a rectangle piece lies inside
    # the domain; one that is not overhangs the boundary and is dropped.
    inside = set(omap[0])
    scat = set().union(*omap[1:]) if len(omap) > 1 else set()
    scat_dt = sorted(inside & scat)
    matx_dt = sorted(inside - scat)

    gmsh.option.setNumber("Mesh.MeshSizeMax", h)
    gmsh.option.setNumber("Mesh.MeshSizeMin", 0.0)
    if scat_dt:
        h_scat = 0.5 * h if h_scat is None else h_scat
        bnd = gmsh.model.getBoundary(scat_dt, combined=False, oriented=False)
        curves = sorted({abs(int(t)) for d, t in bnd if d == 1})
        fd = gmsh.model.mesh.field.add("Distance")
        gmsh.model.mesh.field.setNumbers(fd, "CurvesList", curves)
        ft = gmsh.model.mesh.field.add("Threshold")
        gmsh.model.mesh.field.setNumber(ft, "InField", fd)
        gmsh.model.mesh.field.setNumber(ft, "SizeMin", h_scat)
        gmsh.model.mesh.field.setNumber(ft, "SizeMax", h)
        gmsh.model.mesh.field.setNumber(ft, "DistMin", 0.0)
        gmsh.model.mesh.field.setNumber(ft, "DistMax", refine_dist * h)
        gmsh.model.mesh.field.setAsBackgroundMesh(ft)
        # Let the field alone set the size, or gmsh blends in its own estimates
        # and the scatterer rims silently coarsen.
        for opt in ("Mesh.MeshSizeExtendFromBoundary",
                    "Mesh.MeshSizeFromPoints", "Mesh.MeshSizeFromCurvature"):
            gmsh.option.setNumber(opt, 0)

    gmsh.model.mesh.generate(2)
    npx, index = _read_nodes(gmsh)

    tri_blocks, z_blocks = [], []
    for dt_list, zval in ((scat_dt, 1.0), (matx_dt, 0.0)):
        for _, tag in dt_list:
            etags, ntags = gmsh.model.mesh.getElementsByType(2, tag=tag)
            if etags.size == 0:
                continue
            conn = np.fromiter((index[int(v)] for v in ntags),
                               np.int64, count=ntags.size).reshape(-1, 3)
            tri_blocks.append(conn)
            z_blocks.append(np.full(conn.shape[0], zval))
    gmsh.finalize()

    tris = np.vstack(tri_blocks)
    npx, tris = _prune_nodes(npx, tris)
    return npx, tris, np.concatenate(z_blocks)


def painted_mesh(Lx, Ly, h, shapes):
    """LEGACY, staircased: mesh a plain rectangle, then assign each element to a
    phase by testing its *centroid*.

    This is the construction that produced the seven-order-of-magnitude spread
    documented in the module docstring.  It is kept only as the control against
    which ``conforming_mesh`` is measured, and must not be used for results.
    """
    gmsh = _gmsh_session()
    gmsh.model.add("plain")
    gmsh.model.occ.addRectangle(0, 0, 0, Lx, Ly)
    gmsh.model.occ.synchronize()
    gmsh.option.setNumber("Mesh.MeshSizeMax", h)
    gmsh.option.setNumber("Mesh.MeshSizeMin", 0.7 * h)
    gmsh.model.mesh.generate(2)
    npx, index = _read_nodes(gmsh)
    etags, ntags = gmsh.model.mesh.getElementsByType(2)
    tris = np.fromiter((index[int(v)] for v in ntags),
                       np.int64, count=ntags.size).reshape(-1, 3)
    gmsh.finalize()

    cents = npx[tris].mean(1)
    z = np.zeros(tris.shape[0])
    for sh in shapes:
        if sh.kind == "disk":
            cx, cy, r = sh.data
            z[np.hypot(cents[:, 0] - cx, cents[:, 1] - cy) < r] = 1.0
        else:
            V = sh.data
            hit = np.ones(cents.shape[0], bool)
            for k in range(len(V)):
                e = V[(k + 1) % len(V)] - V[k]
                cross = (e[0] * (cents[:, 1] - V[k][1])
                         - e[1] * (cents[:, 0] - V[k][0]))
                hit &= cross >= 0.0
            z[hit] = 1.0
    return npx, tris, z


# --------------------------------------------------------------------------- #
#  Domain-wall geometry
# --------------------------------------------------------------------------- #
def wall_polyline(Lx, Ly, *, bend=False, seg2_deg=120.0,
                  x_frac=0.40, y0=2.0, margin=2.0):
    """Lattice-aligned (zigzag) domain wall.  Straight: along a2.  Bent: along a2
    to mid-height, then along ``seg2_deg`` (120deg gives the sharp 60deg kink)."""
    start = np.array([x_frac * Lx, y0])
    if not bend:
        return np.array([start, start + ((Ly - margin - start[1]) / A2[1]) * A2])
    corner = start + ((0.5 * Ly - start[1]) / A2[1]) * A2
    th = np.deg2rad(seg2_deg)
    u2 = np.array([np.cos(th), np.sin(th)])
    L2 = ((Ly - margin - corner[1]) / u2[1] if abs(u2[1]) > 1e-3
          else 0.30 * Lx / max(abs(u2[0]), 1e-3))
    return np.array([start, corner, corner + L2 * u2])


def wall_side_fn(poly):
    """``side(xy) -> bool``, True on one side of the (possibly bent) wall.

    Uses the *nearest* segment, so it is correct for multi-segment walls.  The
    straight-wall special case in ``valley_robust`` used a single-segment test;
    for a two-point polyline the two agree exactly.
    """
    poly = np.asarray(poly, float)

    def side(r):
        r = np.asarray(r, float)
        best_d, sgn = np.inf, 1.0
        for k in range(len(poly) - 1):
            p0, seg = poly[k], poly[k + 1] - poly[k]
            t = np.clip(np.dot(r - p0, seg) / np.dot(seg, seg), 0.0, 1.0)
            d = float(np.hypot(*(r - (p0 + t * seg))))
            if d < best_d:
                best_d = d
                sgn = np.sign(seg[0] * (r[1] - p0[1]) - seg[1] * (r[0] - p0[0]))
        return sgn < 0
    return side


def quadratic_sponge(npx, Lx, Ly, *, margin=1.8, strength=25.0):
    """Mass-proportional absorbing layer ramping quadratically into all four
    boundaries (nodal coefficient; multiplied by the lumped mass in ``assemble``)."""
    xs, ys = npx[:, 0], npx[:, 1]
    d = np.maximum.reduce([(margin - xs) / margin, (xs - (Lx - margin)) / margin,
                           (margin - ys) / margin, (ys - (Ly - margin)) / margin])
    return np.maximum(0.0, d) ** 2 * strength


# --------------------------------------------------------------------------- #
#  Scalar (antiplane-shear) harmonic assembly
# --------------------------------------------------------------------------- #
_MREF = np.array([[2.0, 1.0, 1.0],
                  [1.0, 2.0, 1.0],
                  [1.0, 1.0, 2.0]]) / 12.0


def assemble(npx, tris, z, *, muf, omega, eta, sponge,
             mu_m=1.0, rho_f=1.6, rho_m=1.2, penal=3.0):
    r"""Assemble the complex-symmetric dynamic stiffness

        D = (1 + i*eta) K - omega^2 M + i*omega*C_sponge

    for P1 antiplane shear.  ``eta`` is structural damping and *is applied* --
    the defect this module exists to remove.

    Vectorized over elements.  Both phases are isotropic, ``mu_f = muf*I`` and
    ``mu_m = mu_m*I``, and the P1 element stiffness is linear in mu, so the
    two-phase blend ``w*k(mu_f) + (1-w)*k(mu_m)`` equals ``k`` evaluated at the
    blended scalar ``w*muf + (1-w)*mu_m``.  That identity is exact and halves the
    per-element work relative to the loop it replaces.

    Returns ``(D, K, M)``.
    """
    X = npx[tris]                                        # (T,3,2)
    x0, y0 = X[:, 0, 0], X[:, 0, 1]
    x1, y1 = X[:, 1, 0], X[:, 1, 1]
    x2, y2 = X[:, 2, 0], X[:, 2, 1]
    area = 0.5 * np.abs((x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0))

    b = np.stack([y1 - y2, y2 - y0, y0 - y1], axis=1)    # (T,3) = 2A * dN/dx
    c = np.stack([x2 - x1, x0 - x2, x1 - x0], axis=1)    # (T,3) = 2A * dN/dy
    G = ((b[:, :, None] * b[:, None, :] + c[:, :, None] * c[:, None, :])
         / (4.0 * area)[:, None, None])

    w = np.asarray(z, float) ** penal
    kel = (w * muf + (1.0 - w) * mu_m)[:, None, None] * G
    rho = np.asarray(z, float) * rho_f + (1.0 - np.asarray(z, float)) * rho_m
    mel = (rho * area)[:, None, None] * _MREF

    n = npx.shape[0]
    iK = np.kron(tris, np.ones((3, 1), np.int64)).ravel()
    jK = np.kron(tris, np.ones((1, 3), np.int64)).ravel()
    K = sp.csr_matrix((kel.ravel(), (iK, jK)), shape=(n, n))
    M = sp.csr_matrix((mel.ravel(), (iK, jK)), shape=(n, n))

    Md = np.asarray(M.sum(1)).ravel()
    D = ((1.0 + 1j * eta) * K - omega ** 2 * M
         + 1j * omega * sp.diags(sponge * Md))
    return D, K, M


def solve_transmission(npx, tris, z, *, muf, omega, eta, sponge, src, probe,
                       **kw):
    """Point-drive at node ``src``; return ``(u, T)`` with ``T = sum |u|^2`` over
    the boolean node mask ``probe``."""
    D, _, _ = assemble(npx, tris, z, muf=muf, omega=omega, eta=eta,
                       sponge=sponge, **kw)
    F = np.zeros(npx.shape[0], complex)
    F[src] = 1.0
    u = spla.spsolve(D.tocsc(), F)
    return u, float(np.sum(np.abs(u[probe]) ** 2))
