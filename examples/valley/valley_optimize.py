"""
Gradient-based optimization of a topological (valley) band gap, with an honest
accounting of the geometry vs. anisotropy contributions.

Design fields on a fixed, C3-symmetric triangular-rod cell (isolated Dirac at K):
    z(x)      element density  (SIMP two-phase: stiff fiber vs soft matrix)
    theta(x)  fiber orientation (anisotropic mu(theta))

Objective (MAXIMIZED):  the complete gap of the valley (Dirac) band pair,
    J(z,theta) = softmin_k omega_{p+1}(k) - softmax_k omega_p(k),
over a Brillouin-zone k-sample, using the Hellmann--Feynman sensitivities
d omega / d z, d omega / d theta (FD-verified) and MMA.  A mesh filter
regularizes both fields.

We run three modes to isolate the mechanisms:
    theta : optimize orientation only (anisotropy as the Dirac mass),  z fixed
    z     : optimize density only (geometry as the mass),             theta fixed
    both  : optimize both
and report the achievable gap for each -- a transparent measure of how much the
anisotropy actually contributes to the topological gap.
"""
import argparse
import numpy as np
from scipy.spatial import cKDTree
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from wavetopo.valley_cell import triangle_cell
from wavetopo.honeycomb_mesh import HoneycombBloch
from wavetopo.scalar import MaterialSH
from wavetopo.cfrp_optimizer import MMA
from wavetopo.bandgap_opt import _softmax, _softmin


def mesh_filter(cents, R):
    tree = cKDTree(cents); pairs = tree.query_pairs(R, output_type="ndarray")
    n = cents.shape[0]
    rows = list(range(n)); cols = list(range(n)); w = [R]*n
    for i, j in pairs:
        d = np.hypot(*(cents[i]-cents[j])); wij = R-d
        rows += [i, j]; cols += [j, i]; w += [wij, wij]
    import scipy.sparse as sp
    W = sp.csr_matrix((w, (rows, cols)), shape=(n, n))
    return sp.diags(1.0/np.asarray(W.sum(1)).ravel()) @ W


class ValleyGapOpt:
    def __init__(self, hb, ksample, p=1, rho_ks=20.0, Rf=0.12, c_vol=8.0):
        self.hb = hb; self.ks = ksample; self.p = p; self.rho = rho_ks
        self.cents = hb.m["npx"][hb.m["tris"]].mean(1)
        self.area = self._areas()
        self.P = mesh_filter(self.cents, Rf)
        self.Nt = hb.m["tris"].shape[0]
        self.vtarget = float(np.sum(self.area * hb.zfib) / self.area.sum())
        self.c_vol = c_vol

    def _areas(self):
        npx, tris = self.hb.m["npx"], self.hb.m["tris"]
        a = np.zeros(tris.shape[0])
        for e, tri in enumerate(tris):
            (x0, y0), (x1, y1), (x2, y2) = npx[tri]
            a[e] = 0.5*abs((x1-x0)*(y2-y0)-(x2-x0)*(y1-y0))
        return a

    def evaluate(self, zc, thc, grad=True):
        z = self.P @ zc; th = self.P @ thc
        self.hb.assemble(z, th)
        p = self.p
        lo, hi = [], []
        dlo_z, dhi_z, dlo_t, dhi_t = [], [], [], []
        for (kx, ky) in self.ks:
            if grad:
                w, dz, dt = self.hb.eigen_sensitivity(kx, ky, p+2)
                dlo_z.append(dz[p]); dhi_z.append(dz[p+1])
                dlo_t.append(dt[p]); dhi_t.append(dt[p+1])
            else:
                w = self.hb.bands_at_k(kx, ky, p+2)
            lo.append(w[p]); hi.append(w[p+1])
        lo = np.array(lo); hi = np.array(hi)
        smax_lo, wlo = _softmax(lo, self.rho)
        smin_hi, whi = _softmin(hi, self.rho)
        gap = smin_hi - smax_lo
        vol = float(np.sum(self.area*z)/self.area.sum())
        obj = gap - self.c_vol*(vol-self.vtarget)**2      # maximized
        if not grad:
            return obj, gap, vol
        dz = self.P.T @ (whi @ np.array(dhi_z) - wlo @ np.array(dlo_z))
        dt = self.P.T @ (whi @ np.array(dhi_t) - wlo @ np.array(dlo_t))
        dvol = self.area/self.area.sum()
        dz = dz - self.c_vol*2*(vol-self.vtarget)*(self.P.T @ dvol)
        return obj, gap, vol, dz, dt


def run(mode, opt, iters, move=0.06):
    Nt = opt.Nt
    z0 = opt.hb.zfib.copy(); t0 = np.zeros(Nt)
    zc, tc = z0.copy(), t0.copy()
    # design masks per mode
    do_z = mode in ("z", "both"); do_t = mode in ("theta", "both")
    xs, dv = [], []
    if do_z:
        xs.append(zc); dv.append("z")
    if do_t:
        xs.append(tc); dv.append("t")
    x = np.concatenate(xs)
    lo = np.concatenate([(-np.pi/2 if d == "t" else 0.0)*np.ones(Nt) for d in dv])
    hi = np.concatenate([(np.pi/2 if d == "t" else 1.0)*np.ones(Nt) for d in dv])
    mma = MMA(lo, hi, move=move)
    g_init, gi, _ = opt.evaluate(zc, tc, grad=False)
    hist = []; best = (gi, zc.copy(), tc.copy())     # running best FEASIBLE gap
    for it in range(iters):
        obj, gap, vol, dz, dt = opt.evaluate(zc, tc, grad=True)
        if gap > best[0] and abs(vol-opt.vtarget) < 0.03:
            best = (gap, zc.copy(), tc.copy())
        hist.append(best[0])                          # monotone best-feasible
        g = []
        if do_z:
            g.append(-dz)          # MMA minimizes; maximize obj -> minimize -obj
        if do_t:
            g.append(-dt)
        xn = mma.update(x, np.concatenate(g))
        off = 0
        if do_z:
            zc = np.clip(xn[off:off+Nt], 0, 1); off += Nt
        if do_t:
            tc = xn[off:off+Nt]
        x = xn
    return best[0], best[1], best[2], hist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h", type=float, default=0.075)
    ap.add_argument("--R", type=float, default=0.40)
    ap.add_argument("--muL", type=float, default=120.0)
    ap.add_argument("--muT", type=float, default=40.0)
    ap.add_argument("--iters", type=int, default=45)
    ap.add_argument("--alpha0", type=float, default=24.0,
                    help="starting rod angle (breaks C3v so gradients flow)")
    ap.add_argument("--nk", type=int, default=7)
    ap.add_argument("--out", default="results/valley_optimize.png")
    args = ap.parse_args()

    mat = MaterialSH(mu_L=args.muL, mu_T=args.muT, mu_m=1.0, rho_f=1.6, rho_m=1.2)
    # slightly symmetry-broken geometry (alpha0<30): a small seed valley gap so
    # the gradient of the gap w.r.t. the design is non-zero (alpha=30 is a saddle)
    m = triangle_cell(args.alpha0, R=args.R, h=args.h)
    hb = HoneycombBloch(m, mat)
    b1, b2 = hb.recip()
    ks = [(s*b1+t*b2)[0:2] for s in np.linspace(0, 1, args.nk, endpoint=False)
          for t in np.linspace(0, 1, args.nk, endpoint=False)]
    ks = [(k[0], k[1]) for k in ks]
    opt = ValleyGapOpt(hb, ks, p=1, c_vol=45.0)

    _, g0, _ = opt.evaluate(hb.zfib.copy(), np.zeros(opt.Nt), grad=False)
    print(f"mesh {hb.Nn} nodes, {opt.Nt} tris, vtarget={opt.vtarget:.3f}; "
          f"initial complete gap (orientation=0 rod) = {g0:+.4f}")
    results = {}
    for mode in ["theta", "z", "both"]:
        gap, zc, tc, hist = run(mode, opt, args.iters)
        results[mode] = (gap, zc, tc, hist)
        print(f"  optimize [{mode:5s}]: best complete valley gap = {gap:+.4f}")

    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    for mode, (gap, zc, tc, hist) in results.items():
        ax[0].plot(hist, label=f"{mode} (gap={gap:.3f})")
    ax[0].axhline(0, color="k", lw=0.6)
    ax[0].set_xlabel("MMA iteration"); ax[0].set_ylabel("complete valley gap")
    ax[0].set_title("Gradient optimization of the topological gap")
    ax[0].legend()
    # bar chart: contribution accounting
    names = ["orientation\n(anisotropy)", "density\n(geometry)", "both"]
    gaps = [results[m][0] for m in ["theta", "z", "both"]]
    ax[1].bar(names, gaps, color=["tab:red", "tab:blue", "tab:purple"])
    ax[1].axhline(g0, color="k", ls="--", lw=1, label=f"start ({g0:.3f})")
    ax[1].set_ylabel("max complete valley gap"); ax[1].legend()
    ax[1].set_title("Geometry vs. anisotropy contribution")
    fig.suptitle("What opens the topological valley gap? honest accounting", y=1.0)
    plt.tight_layout(); fig.savefig(args.out, dpi=140, bbox_inches="tight")
    print("saved", args.out)
    np.savez("results/valley_optimize.npz",
             **{f"{m}_z": results[m][1] for m in results},
             **{f"{m}_t": results[m][2] for m in results},
             gaps=gaps, g0=g0)


if __name__ == "__main__":
    main()
