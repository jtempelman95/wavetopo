r"""
In-plane orthotropic CFRP on a CONFORMING triangular mesh, for co-designed
topology + toolpath wave-and-stiffness problems.

Everything vector (in-plane elastic) in this repo was on a structured quad grid;
everything on conforming gmsh triangles was scalar (antiplane).  This module is
the missing piece: constant-strain-triangle (CST) plane-stress elements carrying
the same rotated orthotropic fiber stiffness Df(theta) as wavetopo/cfrp.py, on a
gmsh mesh that conforms to prescribed through-holes.

Reused from cfrp.py (mesh-independent): the Material (Qf, Dm) and the exact
five-term Fourier decomposition of Df(theta) into 3x3 Voigt coefficient matrices
C_s, so kf(theta) = sum_s c_s(theta) * [A B^T C_s B].  Only the element assembly
(CST here, quad there) differs.

Design fields:
  z : element density (topology), SIMP  s(z) = eps + (1-eps) z^p
  theta : element fiber orientation (toolpath)
Element stiffness  k_e = s(z_e) * kf(theta_e);  mass  m_e = (rho_min + z_e(1-rho_min)) rho_f m_e^0.
Holes are CUT from the geometry (traction-free rims), not represented as soft
density -- the distinction the paper argues for throughout.
"""
from __future__ import annotations

from collections import namedtuple

import numpy as np
import scipy.sparse as sp

from .cfrp import Material, Df_of_theta

__all__ = ["Material", "conforming_plate", "CSTFourier", "PlateMesh"]

PlateMesh = namedtuple("PlateMesh",
                       "npx tris cx cy area edof ndof nnode Lx Ly holes")


# --------------------------------------------------------------------------- #
#  Conforming mesh: rectangle with circular through-holes cut out
# --------------------------------------------------------------------------- #
def conforming_plate(Lx, Ly, h, holes=(), *, h_hole=None, refine_dist=1.5):
    """Triangulate [0,Lx]x[0,Ly] with the mesh conforming to each circular hole
    ``(cx, cy, r)`` -- the disk is removed from the domain, giving an exactly
    circular traction-free rim (natural BC), refined to ``h_hole`` (default h/2)
    on the rim and grading back to ``h`` over ``refine_dist`` rim-gaps.
    """
    import gmsh
    if gmsh.isInitialized():
        gmsh.finalize()
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("plate")
    occ = gmsh.model.occ

    rect = occ.addRectangle(0, 0, 0, Lx, Ly)
    holes = [tuple(map(float, hxy)) for hxy in holes]
    if holes:
        disks = [(2, occ.addDisk(cx, cy, 0, r, r)) for cx, cy, r in holes]
        occ.cut([(2, rect)], disks)
    occ.synchronize()

    surfs = [s[1] for s in gmsh.model.getEntities(2)]
    gmsh.model.addPhysicalGroup(2, surfs, 1)

    gmsh.option.setNumber("Mesh.MeshSizeMax", h)
    gmsh.option.setNumber("Mesh.MeshSizeMin", h)
    if holes:
        h_hole = 0.5 * h if h_hole is None else h_hole
        bnd = gmsh.model.getBoundary([(2, s) for s in surfs], combined=False,
                                     oriented=False)
        # rim curves = boundary curves that are not the four rectangle sides
        rim = []
        for dim, tag in bnd:
            if dim != 1:
                continue
            x, y, _ = gmsh.model.occ.getCenterOfMass(1, abs(tag))
            on_side = (abs(x) < 1e-6 or abs(x - Lx) < 1e-6
                       or abs(y) < 1e-6 or abs(y - Ly) < 1e-6)
            if not on_side:
                rim.append(abs(tag))
        if rim:
            fd = gmsh.model.mesh.field.add("Distance")
            gmsh.model.mesh.field.setNumbers(fd, "CurvesList", rim)
            ft = gmsh.model.mesh.field.add("Threshold")
            gmsh.model.mesh.field.setNumber(ft, "InField", fd)
            gmsh.model.mesh.field.setNumber(ft, "SizeMin", h_hole)
            gmsh.model.mesh.field.setNumber(ft, "SizeMax", h)
            gmsh.model.mesh.field.setNumber(ft, "DistMin", 0.0)
            gmsh.model.mesh.field.setNumber(ft, "DistMax", refine_dist * h)
            gmsh.model.mesh.field.setAsBackgroundMesh(ft)
            for opt in ("Mesh.MeshSizeExtendFromBoundary",
                        "Mesh.MeshSizeFromPoints", "Mesh.MeshSizeFromCurvature"):
                gmsh.option.setNumber(opt, 0)

    gmsh.model.mesh.generate(2)
    tags, coord, _ = gmsh.model.mesh.getNodes()
    npx = coord.reshape(-1, 3)[:, :2]
    index = {int(t): i for i, t in enumerate(tags)}
    etags, ntags = gmsh.model.mesh.getElementsByType(2)
    tris = np.fromiter((index[int(v)] for v in ntags), np.int64,
                       count=ntags.size).reshape(-1, 3)
    gmsh.finalize()

    # prune unreferenced nodes and reindex
    used = np.zeros(npx.shape[0], bool)
    used[tris.ravel()] = True
    if not used.all():
        remap = np.full(npx.shape[0], -1, np.int64)
        remap[used] = np.arange(int(used.sum()))
        npx, tris = npx[used], remap[tris]

    X = npx[tris]
    cx, cy = X[:, :, 0].mean(1), X[:, :, 1].mean(1)
    x0, y0 = X[:, 0, 0], X[:, 0, 1]
    x1, y1 = X[:, 1, 0], X[:, 1, 1]
    x2, y2 = X[:, 2, 0], X[:, 2, 1]
    area = 0.5 * np.abs((x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0))
    n = npx.shape[0]
    edof = np.empty((tris.shape[0], 6), np.int64)
    edof[:, 0::2] = 2 * tris
    edof[:, 1::2] = 2 * tris + 1
    return PlateMesh(npx, tris, cx, cy, area, edof, 2 * n, n, Lx, Ly, holes)


# --------------------------------------------------------------------------- #
#  CST plane-stress element with the Fourier orientation decomposition
# --------------------------------------------------------------------------- #
def _cst_B(X):
    """Vectorized CST strain-displacement B (T,3,6) and area (T,).
    X is (T,3,2); dof order per element is [u0,v0,u1,v1,u2,v2]."""
    x0, y0 = X[:, 0, 0], X[:, 0, 1]
    x1, y1 = X[:, 1, 0], X[:, 1, 1]
    x2, y2 = X[:, 2, 0], X[:, 2, 1]
    det = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)
    A = 0.5 * np.abs(det)
    inv = 1.0 / det                                     # signed
    b = np.stack([y1 - y2, y2 - y0, y0 - y1], 1) * inv[:, None]   # dN/dx
    c = np.stack([x2 - x1, x0 - x2, x1 - x0], 1) * inv[:, None]   # dN/dy
    T = X.shape[0]
    B = np.zeros((T, 3, 6))
    B[:, 0, 0::2] = b
    B[:, 1, 1::2] = c
    B[:, 2, 0::2] = c
    B[:, 2, 1::2] = b
    return B, A


_MREF6 = np.array([[2, 0, 1, 0, 1, 0],
                   [0, 2, 0, 1, 0, 1],
                   [1, 0, 2, 0, 1, 0],
                   [0, 1, 0, 2, 0, 1],
                   [1, 0, 1, 0, 2, 0],
                   [0, 1, 0, 1, 0, 2]], float) / 12.0


class CSTFourier:
    """Per-triangle Fourier stiffness: kf(theta) = sum_s c_s(theta) kf_s, with
    kf_s = A * B^T C_s B (T,6,6) built from the mesh-independent C_s of Df(theta).
    Also carries the matrix-phase km and consistent mass Me."""

    def __init__(self, mesh: PlateMesh, material: Material, nsamp=720):
        self.mesh, self.mat = mesh, material
        X = mesh.npx[mesh.tris]
        B, A = _cst_B(X)
        self.B, self.A = B, A

        # --- mesh-independent Fourier fit of Df(theta) -> C_s (5,3,3) ------ #
        th = np.linspace(-np.pi, np.pi, nsamp, endpoint=False)
        basis = self._basis(th)                              # (nsamp,5)
        comps = np.stack([Df_of_theta(t, material.Qf) for t in th])  # (nsamp,3,3)
        coef, *_ = np.linalg.lstsq(basis, comps.reshape(nsamp, 9), rcond=None)
        C = coef.reshape(5, 3, 3)
        self.fit_err = np.max(np.abs((basis @ coef).reshape(nsamp, 3, 3) - comps)) \
            / np.max(np.abs(comps))

        # kf_s[s] = A * B^T C_s B  -> (5,T,6,6)
        self.kf_s = np.stack([
            A[:, None, None] * np.einsum('tip,ij,tjq->tpq', B, C[s], B)
            for s in range(5)])
        self.km = (A[:, None, None]
                   * np.einsum('tip,ij,tjq->tpq', B, material.Dm, B))  # (T,6,6)
        self.Me0 = A[:, None, None] * _MREF6                            # (T,6,6)

    @staticmethod
    def _basis(theta):
        theta = np.atleast_1d(np.asarray(theta, float))
        return np.stack([np.ones_like(theta), np.cos(2 * theta),
                         np.sin(2 * theta), np.cos(4 * theta),
                         np.sin(4 * theta)], -1)

    @staticmethod
    def _dbasis(theta):
        theta = np.atleast_1d(np.asarray(theta, float))
        z = np.zeros_like(theta)
        return np.stack([z, -2 * np.sin(2 * theta), 2 * np.cos(2 * theta),
                         -4 * np.sin(4 * theta), 4 * np.cos(4 * theta)], -1)

    def kf(self, theta):
        return np.einsum('ts,stij->tij', self._basis(theta), self.kf_s)

    def dkf(self, theta):
        return np.einsum('ts,stij->tij', self._dbasis(theta), self.kf_s)

    # --- global assembly ------------------------------------------------- #
    def _scatter(self, kel):
        m = self.mesh
        iK = np.repeat(m.edof, 6, axis=1).ravel()
        jK = np.tile(m.edof, (1, 6)).ravel()
        return sp.csr_matrix((kel.ravel(), (iK, jK)),
                             shape=(m.ndof, m.ndof))

    def stiffness(self, z, theta, *, penal=3.0, eps=1e-6):
        s = eps + (1.0 - eps) * np.asarray(z, float) ** penal
        return self._scatter(s[:, None, None] * self.kf(theta))

    def mass(self, z, *, rho_min=1e-6):
        rho = (rho_min + np.asarray(z, float) * (1.0 - rho_min)) * self.mat.rho_f
        return self._scatter(rho[:, None, None] * self.Me0)


# --------------------------------------------------------------------------- #
#  Coupled harmonic (path energy) + static (compliance) problem
# --------------------------------------------------------------------------- #
class CFRPWavePlate:
    r"""Co-design of density z and orientation theta on the conforming plate for
    two objectives sharing one K(z,theta):

      DYNAMIC   D u = F,   D = (1+i eta) K - omega^2 M + i omega C_sponge
                path energy  E = u^H W u   (W signed: +1 in the corridor tube,
                -w_leak in the rest of the interior); MAXIMIZED.
      STATIC    K u_s = Fs   (bcs),  compliance  c = Fs^T u_s;  MINIMIZED.

    Gradients w.r.t. per-element (z, theta) via the discrete adjoint; the harmonic
    system is complex-symmetric so its adjoint solves conj(D) lam = W u.
    """

    def __init__(self, mesh: PlateMesh, material: Material, *, omega, eta=0.02,
                 penal=3.0, eps=1e-6):
        self.mesh, self.mat = mesh, material
        self.fs = CSTFourier(mesh, material)
        self.omega, self.eta, self.penal, self.eps = omega, eta, penal, eps
        self.F = np.zeros(mesh.ndof, complex)
        self.C = sp.csr_matrix((mesh.ndof, mesh.ndof))
        self.W = sp.csr_matrix((mesh.ndof, mesh.ndof))
        self.Fs = np.zeros(mesh.ndof)
        self.fixed = np.array([], int)
        self._free = np.arange(mesh.ndof)
        self._sfree = np.arange(mesh.ndof)

    # ---- setup ---------------------------------------------------------- #
    def set_sponge(self, c_elem):
        self.C = self.fs._scatter(np.asarray(c_elem, float)[:, None, None]
                                  * self.fs.Me0)

    def set_drive(self, F):
        self.F = np.asarray(F, complex)

    def set_weight(self, w_node):
        """Signed nodal weight for E = u^H W u (per DOF, x and y summed)."""
        self.W = sp.diags(np.repeat(w_node, 2)).tocsr()

    def set_static(self, Fs, fixed):
        self.Fs = np.asarray(Fs, float)
        self.fixed = np.unique(np.asarray(fixed, int))
        self._sfree = np.setdiff1d(np.arange(self.mesh.ndof), self.fixed)

    # Two-phase interpolation: z=1 stiff fiber, z=0 soft matrix (PRESENT, not
    # air).  This is essential for the dynamic problem -- a SIMP fiber-vs-void
    # model gives void elements vanishing stiffness AND mass, so the dynamic
    # matrix goes near-singular there and the harmonic response resonates/blows
    # up.  With a matrix phase the wave propagates everywhere (slower in matrix,
    # guided by fiber) and the topology stays a well-posed fiber/matrix layout.
    def _w(self, z):
        return np.asarray(z, float) ** self.penal            # fiber fraction

    def _dw(self, z):
        return self.penal * np.asarray(z, float) ** (self.penal - 1.0)

    # ---- assemble ------------------------------------------------------- #
    def _assemble(self, z, theta):
        w = self._w(z)
        kf = self.fs.kf(theta)
        kel = w[:, None, None] * kf + (1.0 - w)[:, None, None] * self.fs.km
        K = self.fs._scatter(kel)
        z = np.asarray(z, float)
        rho = z * self.mat.rho_f + (1.0 - z) * self.mat.rho_m
        M = self.fs._scatter(rho[:, None, None] * self.fs.Me0)
        return K, M, w, kf

    # ---- dynamic: path energy + gradient -------------------------------- #
    def dynamic(self, z, theta, want_grad=True):
        m = self.mesh
        K, M, w, kf = self._assemble(z, theta)
        D = ((1 + 1j * self.eta) * K - self.omega**2 * M
             + 1j * self.omega * self.C).tocsc()
        u = np.zeros(m.ndof, complex)
        fr = self._free
        u[fr] = sp.linalg.spsolve(D[fr][:, fr], self.F[fr])
        E = float(np.real(np.vdot(u, self.W @ u)))
        if not want_grad:
            return E, u
        # adjoint: conj(D) lam = W u   (E = u^H W u, W real symmetric)
        lam = np.zeros(m.ndof, complex)
        rhs = self.W @ u
        lam[fr] = sp.linalg.spsolve(D.conj()[fr][:, fr], rhs[fr])
        uE = u[m.edof]; lamE = lam[m.edof]
        dkf = self.fs.dkf(theta)
        fac = (1 + 1j * self.eta)
        # dE/dtheta = -2 Re( lam_e^H (1+i eta) w dkf u_e )
        dth = -2.0 * (w * fac * np.einsum('ep,epq,eq->e',
                                          np.conj(lamE), dkf, uE)).real
        # two-phase dD/dz = (1+i eta) w'(z)(kf-km) - omega^2 (rho_f-rho_m) Me0
        eK = np.einsum('ep,epq,eq->e', np.conj(lamE), kf - self.fs.km, uE) * fac
        eM = np.einsum('ep,epq,eq->e', np.conj(lamE), self.fs.Me0, uE)
        dz = -2.0 * (self._dw(z) * eK
                     - self.omega**2 * (self.mat.rho_f - self.mat.rho_m)
                     * eM).real
        return E, u, dz, dth

    # ---- static: compliance + gradient ---------------------------------- #
    def static(self, z, theta, want_grad=True):
        m = self.mesh
        w = self._w(z); kf = self.fs.kf(theta)
        kel = w[:, None, None] * kf + (1.0 - w)[:, None, None] * self.fs.km
        K = self.fs._scatter(kel).tocsc()
        u = np.zeros(m.ndof)
        fr = self._sfree
        u[fr] = sp.linalg.spsolve(K[fr][:, fr], self.Fs[fr])
        c = float(self.Fs @ u)
        if not want_grad:
            return c, u
        uE = u[m.edof]
        dkf = self.fs.dkf(theta)
        # self-adjoint: dc/dp = -u^T dK/dp u;  dK/dz = w'(z)(kf-km), dK/dth = w dkf
        dth = -(w * np.einsum('ep,epq,eq->e', uE, dkf, uE))
        dz = -(self._dw(z) * np.einsum('ep,epq,eq->e', uE, kf - self.fs.km, uE))
        return c, u, dz, dth

    def volume(self, z):
        A = self.mesh.area
        return float((A @ z) / A.sum())


def cone_filter(mesh: PlateMesh, radius):
    """Area-weighted cone (linear) density filter on element centroids, returned
    as a sparse (T,T) row-normalized matrix -- the unstructured-mesh analogue of
    the structured cone filter, so a minimum length scale is imposed on z."""
    from scipy.spatial import cKDTree
    c = np.column_stack([mesh.cx, mesh.cy])
    A = mesh.area
    tree = cKDTree(c)
    pairs = tree.query_ball_point(c, radius)
    rows, cols, vals = [], [], []
    for i, nbr in enumerate(pairs):
        nbr = np.asarray(nbr, int)
        w = np.maximum(0.0, radius - np.linalg.norm(c[nbr] - c[i], axis=1)) * A[nbr]
        rows.append(np.full(nbr.size, i)); cols.append(nbr); vals.append(w / w.sum())
    return sp.csr_matrix((np.concatenate(vals),
                          (np.concatenate(rows), np.concatenate(cols))),
                         shape=(len(A), len(A)))
