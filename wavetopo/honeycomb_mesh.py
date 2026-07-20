"""
Periodic, C3-clean unstructured honeycomb unit-cell mesh via gmsh.

A structured/sheared mesh systematically breaks the C3 symmetry that protects
the honeycomb Dirac cone (the splitting does not vanish under refinement).  An
unstructured (Delaunay) mesh has only random, mesh-convergent anisotropy, so the
Dirac gap -> 0 as the mesh refines -- the correct setting for valley physics.

We mesh the rhombic primitive cell (lattice vectors a1, a2 at 60 deg) with two
conforming disk inclusions at the A and B sublattice sites, enforce periodic
boundary node matching on both edge pairs, and return everything needed for a
scalar (antiplane-shear) P1 Bloch calculation:

    npx      (Nn,2) node coordinates
    tris     (Nt,3) triangle connectivity (0-based)
    zfib     (Nt,)  1.0 inside a disk (fiber), else 0.0
    master   (Nn,)  reduced-DOF index of each node's Bloch master
    shift    (Nn,2) lattice shift from node to its master  (phase e^{i k.shift})
    a1, a2, sites
"""
from __future__ import annotations

import numpy as np


def honeycomb_cell(rA=0.34, rB=0.34, b=1.0, h=0.06, verbose=False):
    """Generate one periodic rhombic honeycomb cell. h = target mesh size."""
    import gmsh
    s3 = np.sqrt(3.0)
    a1 = b * np.array([1.5, s3 / 2])
    a2 = b * np.array([1.5, -s3 / 2])
    A = (a1 + a2) / 3.0                     # sublattice A site  (=(b,0))
    B = 2.0 * (a1 + a2) / 3.0               # sublattice B site  (=(2b,0))

    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 1 if verbose else 0)
    gmsh.model.add("hc")
    occ = gmsh.model.occ
    # parallelogram
    P = [occ.addPoint(0, 0, 0), occ.addPoint(*a1, 0),
         occ.addPoint(*(a1 + a2), 0), occ.addPoint(*a2, 0)]
    L = [occ.addLine(P[0], P[1]), occ.addLine(P[1], P[2]),
         occ.addLine(P[2], P[3]), occ.addLine(P[3], P[0])]
    cell = occ.addPlaneSurface([occ.addCurveLoop(L)])
    dA = occ.addDisk(A[0], A[1], 0, rA, rA)
    dB = occ.addDisk(B[0], B[1], 0, rB, rB)
    # fragment: conform mesh to disk boundaries; keep material regions
    frag, _ = occ.fragment([(2, cell)], [(2, dA), (2, dB)])
    occ.synchronize()

    # classify resulting surfaces as fiber (disk) or matrix by centroid
    fiber_surf, matrix_surf = [], []
    for (dim, tag) in gmsh.model.getEntities(2):
        cx, cy, _ = gmsh.model.occ.getCenterOfMass(2, tag)
        if (np.hypot(cx - A[0], cy - A[1]) < rA * 1.05 or
                np.hypot(cx - B[0], cy - B[1]) < rB * 1.05):
            fiber_surf.append(tag)
        else:
            matrix_surf.append(tag)

    # identify the four outer boundary edges (after fragment) by midpoint
    def edge_mid(tag):
        return np.array(gmsh.model.occ.getCenterOfMass(1, tag)[:2])
    outer = {}
    for (dim, tag) in gmsh.model.getEntities(1):
        m = edge_mid(tag)
        # classify against the 4 parallelogram edges by point-line distance
        for name, p0, d in [("bot", np.zeros(2), a1), ("top", a2, a1),
                            ("left", np.zeros(2), a2), ("right", a1, a2)]:
            t = np.dot(m - p0, d) / np.dot(d, d)
            proj = p0 + t * d
            if 0 <= t <= 1 and np.hypot(*(m - proj)) < 1e-6:
                outer.setdefault(name, []).append(tag)

    def tmat(t):
        T = np.eye(4); T[0, 3], T[1, 3] = t[0], t[1]
        return T.flatten().tolist()
    gmsh.model.mesh.setPeriodic(1, outer["right"], outer["left"], tmat(a1))
    gmsh.model.mesh.setPeriodic(1, outer["top"], outer["bot"], tmat(a2))

    gmsh.option.setNumber("Mesh.MeshSizeMax", h)
    gmsh.option.setNumber("Mesh.MeshSizeMin", h * 0.6)
    gmsh.model.mesh.generate(2)

    # ---- extract mesh ------------------------------------------------- #
    ntag, ncoord, _ = gmsh.model.mesh.getNodes()
    ncoord = ncoord.reshape(-1, 3)[:, :2]
    tag2idx = {int(t): i for i, t in enumerate(ntag)}
    npx = ncoord.copy()

    # triangles + per-surface material
    zfib = []
    tri_list = []
    for surf in fiber_surf + matrix_surf:
        ety, enodes = gmsh.model.mesh.getElementsByType(2, surf)[0], \
            gmsh.model.mesh.getElementsByType(2, surf)[1]
        conn = enodes.reshape(-1, 3)
        for tri in conn:
            tri_list.append([tag2idx[int(t)] for t in tri])
            zfib.append(1.0 if surf in fiber_surf else 0.0)
    tris = np.array(tri_list, np.int64)
    zfib = np.array(zfib)

    # ---- periodic node correspondence -> Bloch master map ------------- #
    slave_a1, slave_a2 = {}, {}
    for tag in outer["right"]:
        _, sl, ma, _ = gmsh.model.mesh.getPeriodicNodes(1, tag)
        for s, m in zip(sl, ma):
            slave_a1[int(s)] = int(m)
    for tag in outer["top"]:
        _, sl, ma, _ = gmsh.model.mesh.getPeriodicNodes(1, tag)
        for s, m in zip(sl, ma):
            slave_a2[int(s)] = int(m)
    gmsh.finalize()

    Nn = npx.shape[0]
    shift = np.zeros((Nn, 2))
    master_tagidx = np.arange(Nn)
    # resolve each node to its master via a1/a2 chains
    idx2tag = {i: int(t) for t, i in tag2idx.items()}
    for i in range(Nn):
        tag = idx2tag[i]; sh = np.zeros(2)
        for _ in range(4):
            moved = False
            if tag in slave_a1:
                tag = slave_a1[tag]; sh = sh + a1; moved = True
            if tag in slave_a2:
                tag = slave_a2[tag]; sh = sh + a2; moved = True
            if not moved:
                break
        master_tagidx[i] = tag2idx[tag]
        shift[i] = sh
    # reduced index numbering over the master nodes actually used
    uniq = np.unique(master_tagidx)
    remap = {int(m): r for r, m in enumerate(uniq)}
    master = np.array([remap[int(m)] for m in master_tagidx])
    n_red = len(uniq)

    return dict(npx=npx, tris=tris, zfib=zfib, master=master, shift=shift,
                n_red=n_red, a1=a1, a2=a2, A=A, B=B, b=b)


if __name__ == "__main__":
    m = honeycomb_cell(0.34, 0.34, h=0.09)
    print("nodes", m["npx"].shape[0], "tris", m["tris"].shape[0],
          "reduced DOF", m["n_red"], "fiber frac", m["zfib"].mean())
    # sanity: periodic masters should number nodes minus one full boundary layer
    print("shift nonzero for", int((np.abs(m["shift"]).sum(1) > 1e-9).sum()),
          "boundary nodes")


# ====================================================================== #
#  Scalar (antiplane-shear) Bloch solver on the gmsh honeycomb mesh       #
# ====================================================================== #
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import scipy.linalg as sla
from .rhombic import _tri_scalar
from .scalar import MaterialSH


class HoneycombBloch:
    def __init__(self, mesh, material: MaterialSH, *, penal=3.0):
        self.m = mesh
        self.mat = material
        self.penal = penal
        npx, tris = mesh["npx"], mesh["tris"]
        self.Nn = npx.shape[0]
        self.n_red = mesh["n_red"]
        self.a1, self.a2 = mesh["a1"], mesh["a2"]
        # per-triangle basis element matrices for mu(theta)=aI+b[[c2,s2],[s2,-c2]]
        muL, muT, mum = material.mu_L, material.mu_T, material.mu_m
        aC, bC = (muL + muT) / 2, (muL - muT) / 2
        Cs = [aC * np.eye(2), bC * np.diag([1.0, -1.0]),
              bC * np.array([[0.0, 1.0], [1.0, 0.0]])]
        Nt = tris.shape[0]
        self.kf_s = np.zeros((Nt, 3, 3, 3)); self.km = np.zeros((Nt, 3, 3))
        self.Me = np.zeros((Nt, 3, 3))
        mref = np.array([[2.0, 1, 1], [1, 2, 1], [1, 1, 2]]) / 12.0
        for e, tri in enumerate(tris):
            xy = npx[tri]
            for s in range(3):
                self.kf_s[e, s], area = _tri_scalar(xy, Cs[s])
            self.km[e], _ = _tri_scalar(xy, mum * np.eye(2))
            self.Me[e] = mref * area
        self.iK = np.kron(tris, np.ones((3, 1), np.int64)).ravel()
        self.jK = np.kron(tris, np.ones((1, 3), np.int64)).ravel()
        self.zfib = mesh["zfib"]

    def kf(self, theta):
        c = np.stack([np.ones_like(theta), np.cos(2*theta), np.sin(2*theta)], -1)
        return np.einsum('es,esij->eij', c, self.kf_s)

    def dkf(self, theta):
        c = np.stack([np.zeros_like(theta), -2*np.sin(2*theta),
                      2*np.cos(2*theta)], -1)
        return np.einsum('es,esij->eij', c, self.kf_s)

    def eigen_sensitivity(self, kx, ky, n_bands=6):
        """Hellmann--Feynman sensitivities of the lowest n_bands at k:
        d omega / d z_e (per triangle) and d omega / d theta_e.
        Two-phase: K_e = z^p kf(theta) + (1-z^p) km,  M_e = (z rho_f +
        (1-z) rho_m) Me.  Returns (omega, dz[nb,Nt], dtheta[nb,Nt])."""
        w, w2, V, T = self.bands_at_k(kx, ky, n_bands, return_vec=True)
        tris = self.m["tris"]; z, th, p = self._z, self._theta, self.penal
        kf = self.kf(th); dkf = self.dkf(th); km = self.km; Me = self.Me
        drho = self.mat.rho_f - self.mat.rho_m
        dz = np.zeros((n_bands, tris.shape[0]))
        dth = np.zeros((n_bands, tris.shape[0]))
        for b in range(n_bands):
            phi = (T @ V[:, b])[tris]                 # (Nt,3) complex
            c = np.conj(phi)
            e_kfmkm = np.einsum('ei,eij,ej->e', c, kf - km, phi).real
            e_M = np.einsum('ei,eij,ej->e', c, Me, phi).real
            e_dkf = np.einsum('ei,eij,ej->e', c, dkf, phi).real
            dlam_dz = p * z**(p-1) * e_kfmkm - w2[b] * drho * e_M
            dlam_dth = z**p * e_dkf
            sc = 0.5 / max(w[b], 1e-12)
            dz[b] = sc * dlam_dz; dth[b] = sc * dlam_dth
        return w, dz, dth

    def assemble(self, z=None, theta=None):
        z = self.zfib if z is None else z
        Nt = self.m["tris"].shape[0]
        theta = np.zeros(Nt) if theta is None else np.broadcast_to(theta, (Nt,))
        w = z**self.penal
        kel = w[:, None, None]*self.kf(theta) + (1-w)[:, None, None]*self.km
        self._K = sp.csr_matrix((kel.ravel(), (self.iK, self.jK)),
                                shape=(self.Nn, self.Nn))
        rho = z*self.mat.rho_f + (1-z)*self.mat.rho_m
        self._M = sp.csr_matrix(((rho[:, None, None]*self.Me).ravel(),
                                (self.iK, self.jK)), shape=(self.Nn, self.Nn))
        self._z, self._theta = z, theta
        return self._K, self._M

    def T(self, kx, ky):
        sh = self.m["shift"]
        ph = np.exp(1j*(kx*sh[:, 0] + ky*sh[:, 1]))
        return sp.csr_matrix((ph, (np.arange(self.Nn), self.m["master"])),
                             shape=(self.Nn, self.n_red))

    def bands_at_k(self, kx, ky, n_bands=6, return_vec=False):
        T = self.T(kx, ky)
        Kr = (T.conj().T @ self._K @ T).toarray()
        Mr = (T.conj().T @ self._M @ T).toarray()
        Kr = 0.5*(Kr+Kr.conj().T); Mr = 0.5*(Mr+Mr.conj().T)
        w2, V = sla.eigh(Kr, Mr, subset_by_index=[0, n_bands-1])
        w = np.sqrt(np.clip(w2, 0, None))
        return (w, w2, V, T) if return_vec else w

    def recip(self):
        Amat = np.column_stack([self.a1, self.a2])
        Bmat = 2*np.pi*np.linalg.inv(Amat).T
        return Bmat[:, 0], Bmat[:, 1]

    def kpoints(self):
        b1, b2 = self.recip()
        return dict(G=np.zeros(2), K=(2*b1+b2)/3, Kp=(b1+2*b2)/3,
                    M=(b1+b2)/2, b1=b1, b2=b2)
