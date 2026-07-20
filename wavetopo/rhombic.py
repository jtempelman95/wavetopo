"""
Scalar (antiplane-shear) Bloch solver on the PRIMITIVE rhombic honeycomb cell.

The rectangular 4-atom supercell folds the two valleys K and K' onto the same
point, destroying the valley separation needed for valley-Hall edge states.  The
2-atom rhombic primitive cell (lattice vectors a1, a2 at 60 deg) keeps K and K'
as distinct BZ corners -- the correct setting.

We discretise the rhombus with a structured n x n grid in parametric (s,t)
coordinates; each element is a parallelogram with a constant (affine) Jacobian
J = [a1/n, a2/n].  Element matrices use that general Jacobian.

Lattice (bond length b):  a1 = b*(3/2, sqrt3/2), a2 = b*(3/2, -sqrt3/2).
Two atoms: A = (0,0), B = (b, 0).  Stiff anisotropic-fiber disks of radius rA,
rB in a soft matrix; A != B breaks inversion (the valley Dirac mass).
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import scipy.linalg as sla

from .scalar import MaterialSH


def elem_K_J(mu, J):
    """4x4 scalar element matrix int B^T mu B over a parallelogram with constant
    Jacobian J (2x2) mapping reference square [-1,1]^2 -> element.

    The area element is |det J| (positive regardless of the a1,a2 orientation);
    the gradient transform Jinv carries any orientation sign and appears
    quadratically in B^T mu B, so the element matrix is orientation-invariant.
    """
    g = 1 / np.sqrt(3.0)
    gp = [(-g, -g), (g, -g), (g, g), (-g, g)]
    detJ = abs(np.linalg.det(J))
    Jinv = np.linalg.inv(J)
    Ke = np.zeros((4, 4))
    for xi, eta in gp:
        dN = np.array([[-(1 - eta), (1 - eta), (1 + eta), -(1 + eta)],
                       [-(1 - xi), -(1 + xi), (1 + xi), (1 - xi)]]) * 0.25
        B = Jinv @ dN                      # (2,4) physical grads
        Ke += (B.T @ mu @ B) * detJ
    return Ke


def elem_M_J(J):
    g = 1 / np.sqrt(3.0)
    gp = [(-g, -g), (g, -g), (g, g), (-g, g)]
    detJ = abs(np.linalg.det(J))
    Me = np.zeros((4, 4))
    for xi, eta in gp:
        N = 0.25 * np.array([(1 - xi) * (1 - eta), (1 + xi) * (1 - eta),
                             (1 + xi) * (1 + eta), (1 - xi) * (1 + eta)])
        Me += np.outer(N, N) * detJ
    return Me


class RhombicHoneycomb:
    """Scalar Bloch problem on the rhombic primitive honeycomb cell."""

    def __init__(self, n, material: MaterialSH, *, b=1.0, penal=3.0):
        self.n = n
        self.mat = material
        self.penal = penal
        s3 = np.sqrt(3.0)
        self.a1 = b * np.array([1.5, s3 / 2])
        self.a2 = b * np.array([1.5, -s3 / 2])
        self.b = b
        # parametric grid: n x n elements, (n+1)^2 nodes
        self.N = n * n
        self.nnode = (n + 1) ** 2
        # node param coords (i along a1, j along a2), i,j in [0,n]
        self.J = np.column_stack([self.a1 / n, self.a2 / n])   # element Jacobian
        # element centroids in physical coords + parametric
        ec = []
        cents = []
        edof = np.zeros((self.N, 4), np.int64)
        e = 0
        for j in range(n):           # along a2
            for i in range(n):       # along a1
                n1 = j * (n + 1) + i
                edof[e] = [n1, n1 + 1, n1 + 1 + (n + 1), n1 + (n + 1)]
                sc, tc = (i + 0.5) / n, (j + 0.5) / n
                cents.append(sc * self.a1 + tc * self.a2)
                ec.append((sc, tc))
                e += 1
        self.edof = edof
        self.cents = np.array(cents)             # (N,2) physical
        self.iK = np.kron(edof, np.ones((4, 1), np.int64)).ravel()
        self.jK = np.kron(edof, np.ones((1, 4), np.int64)).ravel()
        # node physical coords
        npts = []
        for j in range(n + 1):
            for i in range(n + 1):
                npts.append((i / n) * self.a1 + (j / n) * self.a2)
        self.npx = np.array(npts)
        # Fourier element matrices (scalar mu, general J)
        muL, muT = material.mu_L, material.mu_T
        a, bb = (muL + muT) / 2, (muL - muT) / 2
        Cs = [np.array([[a, 0], [0, a]]), np.array([[bb, 0], [0, -bb]]),
              np.array([[0, bb], [bb, 0]])]
        self.kf_s = np.array([elem_K_J(C, self.J) for C in Cs])
        self.km = elem_K_J(material.mu_m * np.eye(2), self.J)
        self.Me = elem_M_J(self.J)
        self._bloch_index()

    def kf(self, theta):
        c = np.stack([np.ones_like(theta), np.cos(2 * theta),
                      np.sin(2 * theta)], -1)
        return np.einsum('es,sij->eij', c, self.kf_s)

    # honeycomb density: disks at A=(0,0), B=(b,0) (toroidal in the lattice)
    def honey(self, rA, rB):
        z = np.zeros(self.N)
        A = np.array([0.0, 0.0]); B = np.array([self.b, 0.0])
        for center, r in [(A, rA), (B, rB)]:
            for m1 in (-1, 0, 1):
                for m2 in (-1, 0, 1):
                    c = center + m1 * self.a1 + m2 * self.a2
                    d = self.cents - c
                    z[np.hypot(d[:, 0], d[:, 1]) < r] = 1.0
        return z

    def _bloch_index(self):
        n = self.n
        master = np.zeros(self.nnode, np.int64)
        s1 = np.zeros(self.nnode); s2 = np.zeros(self.nnode)
        for j in range(n + 1):
            for i in range(n + 1):
                node = j * (n + 1) + i
                mi, mj = i % n, j % n
                master[node] = mj * n + mi
                s1[node] = 1.0 if i == n else 0.0     # wrapped along a1
                s2[node] = 1.0 if j == n else 0.0     # wrapped along a2
        self._master, self._s1, self._s2 = master, s1, s2
        self.n_red = n * n

    def T(self, kx, ky):
        # phase = exp(i k.a1 * s1 + i k.a2 * s2)
        ph = np.exp(1j * ((kx * self.a1[0] + ky * self.a1[1]) * self._s1
                          + (kx * self.a2[0] + ky * self.a2[1]) * self._s2))
        return sp.csr_matrix((ph, (np.arange(self.nnode), self._master)),
                             shape=(self.nnode, self.n_red))

    def assemble(self, z, theta):
        w = z**self.penal
        kel = w[:, None, None] * self.kf(theta) + (1 - w)[:, None, None] * self.km[None]
        self._K = sp.csr_matrix((kel.ravel(), (self.iK, self.jK)),
                                shape=(self.nnode, self.nnode))
        rho = z * self.mat.rho_f + (1 - z) * self.mat.rho_m
        mel = rho[:, None, None] * self.Me[None]
        self._M = sp.csr_matrix((mel.ravel(), (self.iK, self.jK)),
                                shape=(self.nnode, self.nnode))
        return self._K, self._M

    def bands_at_k(self, kx, ky, n_bands=6, return_vec=False):
        T = self.T(kx, ky)
        Kr = (T.conj().T @ self._K @ T).toarray()
        Mr = (T.conj().T @ self._M @ T).toarray()
        w2, V = sla.eigh(Kr, Mr, subset_by_index=[0, n_bands - 1])
        w = np.sqrt(np.clip(w2, 0, None))
        if return_vec:
            return w, w2, V, T
        return w

    # reciprocal lattice + high-symmetry points (hexagonal BZ)
    def recip(self):
        A = np.column_stack([self.a1, self.a2])
        B = 2 * np.pi * np.linalg.inv(A).T          # columns are b1,b2
        return B[:, 0], B[:, 1]

    def kpoints(self):
        b1, b2 = self.recip()
        G = np.zeros(2)
        # K at corner of hexagonal BZ
        K = (2 * b1 + b2) / 3
        Kp = (b1 + 2 * b2) / 3
        M = (b1 + b2) / 2
        return dict(G=G, K=K, Kp=Kp, M=M, b1=b1, b2=b2)


# ====================================================================== #
#  C3-respecting P1 TRIANGULAR mesh of the rhombic honeycomb cell         #
#  (the sheared-quad mesh breaks C3 and splits the Dirac cone; a          #
#   triangular mesh restores it.)                                         #
# ====================================================================== #
def _tri_scalar(xy, mu):
    """P1 triangle: return (4|A|)^{-1} M^T mu M (2x2->3x3) and area,
    orientation-sign invariant.  xy = 3x2 vertex coords."""
    (x0, y0), (x1, y1), (x2, y2) = xy
    A = 0.5 * ((x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0))
    b = np.array([y1 - y2, y2 - y0, y0 - y1])
    c = np.array([x2 - x1, x0 - x2, x1 - x0])
    Mmat = np.vstack([b, c])                       # (2,3) = [grad] * 2A
    Ke = (Mmat.T @ mu @ Mmat) / (4.0 * abs(A))
    return Ke, abs(A)


class RhombicHoneycombTri:
    """Scalar Bloch problem on the rhombic honeycomb cell, P1 triangles.

    Each parallelogram cell (i,j) is split into two triangles along the
    SW-NE (a1+a2) diagonal, giving a triangular lattice mesh that respects the
    C3 symmetry of the honeycomb -> a clean, mesh-convergent Dirac cone at K.
    Reuses the node grid, Bloch reduction, honeycomb geometry and reciprocal
    lattice of RhombicHoneycomb.
    """

    def __init__(self, n, material: MaterialSH, *, b=1.0, penal=3.0):
        base = RhombicHoneycomb(n, material, b=b, penal=penal)
        self.n, self.b, self.mat, self.penal = n, b, material, penal
        self.a1, self.a2 = base.a1, base.a2
        self.nnode = base.nnode
        self.npx = base.npx
        self._master, self._s1, self._s2 = base._master, base._s1, base._s2
        self.n_red = base.n_red
        self.recip = base.recip
        self.kpoints = base.kpoints

        # triangulate along the SE-NW (a2-a1) diagonal so the mesh nodes form a
        # C3/C6-symmetric triangular lattice (neighbour directions a1, a2,
        # a2-a1 at 60 deg apart) -> restores the honeycomb Dirac cone.  The
        # SW-NE (a1+a2) diagonal instead gives a C2 mesh that splits the cone.
        tris = []
        for j in range(n):
            for i in range(n):
                sw = j * (n + 1) + i
                se = sw + 1
                nw = sw + (n + 1)
                ne = nw + 1
                tris.append([sw, se, nw])          # lower-left
                tris.append([se, ne, nw])          # upper-right
        self.tris = np.array(tris, np.int64)
        self.N = self.tris.shape[0]                # = 2 n^2 triangles
        self.cents = self.npx[self.tris].mean(1)   # (N,2) centroids

        # per-triangle geometry: coefficient element matrices for mu basis
        muL, muT = material.mu_L, material.mu_T
        a_, bb = (muL + muT) / 2, (muL - muT) / 2
        Cs = [np.array([[a_, 0.0], [0.0, a_]]),
              np.array([[bb, 0.0], [0.0, -bb]]),
              np.array([[0.0, bb], [bb, 0.0]])]
        self.kf_s = np.zeros((self.N, 3, 3, 3))    # (tri, basis, node, node)
        self.km = np.zeros((self.N, 3, 3))
        self.Me = np.zeros((self.N, 3, 3))
        mass_ref = np.array([[2.0, 1, 1], [1, 2, 1], [1, 1, 2]]) / 12.0
        for e, tri in enumerate(self.tris):
            xy = self.npx[tri]
            for s in range(3):
                self.kf_s[e, s], area = _tri_scalar(xy, Cs[s])
            self.km[e], _ = _tri_scalar(xy, material.mu_m * np.eye(2))
            self.Me[e] = mass_ref * area
        # assembly index pattern (3 nodes/tri)
        self.iK = np.kron(self.tris, np.ones((3, 1), np.int64)).ravel()
        self.jK = np.kron(self.tris, np.ones((1, 3), np.int64)).ravel()

    def honey(self, rA, rB):
        z = np.zeros(self.N)
        A = np.array([0.0, 0.0]); B = np.array([self.b, 0.0])
        for center, r in [(A, rA), (B, rB)]:
            for m1 in (-1, 0, 1):
                for m2 in (-1, 0, 1):
                    c = center + m1 * self.a1 + m2 * self.a2
                    d = self.cents - c
                    z[np.hypot(d[:, 0], d[:, 1]) < r] = 1.0
        return z

    def kf(self, theta):
        c = np.stack([np.ones_like(theta), np.cos(2 * theta),
                      np.sin(2 * theta)], -1)                # (N,3)
        return np.einsum('es,esij->eij', c, self.kf_s)

    def dkf(self, theta):
        c = np.stack([np.zeros_like(theta), -2 * np.sin(2 * theta),
                      2 * np.cos(2 * theta)], -1)
        return np.einsum('es,esij->eij', c, self.kf_s)

    def assemble(self, z, theta):
        theta = np.broadcast_to(theta, (self.N,)) if np.ndim(theta) == 0 \
            else np.asarray(theta)
        if theta.shape[0] != self.N:               # allow scalar-per-cell input
            theta = np.zeros(self.N)
        w = z**self.penal
        kel = w[:, None, None] * self.kf(theta) + (1 - w)[:, None, None] * self.km
        self._K = sp.csr_matrix((kel.ravel(), (self.iK, self.jK)),
                                shape=(self.nnode, self.nnode))
        rho = z * self.mat.rho_f + (1 - z) * self.mat.rho_m
        mel = rho[:, None, None] * self.Me
        self._M = sp.csr_matrix((mel.ravel(), (self.iK, self.jK)),
                                shape=(self.nnode, self.nnode))
        self._theta, self._z = theta, z
        return self._K, self._M

    def T(self, kx, ky):
        ph = np.exp(1j * ((kx * self.a1[0] + ky * self.a1[1]) * self._s1
                          + (kx * self.a2[0] + ky * self.a2[1]) * self._s2))
        return sp.csr_matrix((ph, (np.arange(self.nnode), self._master)),
                             shape=(self.nnode, self.n_red))

    def bands_at_k(self, kx, ky, n_bands=6, return_vec=False):
        T = self.T(kx, ky)
        Kr = (T.conj().T @ self._K @ T).toarray()
        Mr = (T.conj().T @ self._M @ T).toarray()
        Kr = 0.5 * (Kr + Kr.conj().T); Mr = 0.5 * (Mr + Mr.conj().T)
        w2, V = sla.eigh(Kr, Mr, subset_by_index=[0, n_bands - 1])
        w = np.sqrt(np.clip(w2, 0, None))
        if return_vec:
            return w, w2, V, T
        return w
