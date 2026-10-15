r"""Does wave ENERGY follow the fibres?  Energy-flux vs fibre-director alignment in the
optimized lens, at the design frequency and across the band.

Time-averaged elastic energy flux of the time-harmonic field (e^{+i omega t}, the
solver's convention):  P_j = -1/2 Re(sigma_ij conj(v_i)) = (omega/2)[sigma_ij(u_r) u_i,i
- sigma_ij(u_i) u_r,i].  (Alignment angles are sign-independent.)
(sigma = D_f(theta) eps, Voigt; structural damping neglected in the flux).
Per interior cell we measure the angle delta between P and the fibre director
(mod pi) and report the |P|-weighted mean of delta and the |P|-weighted fraction of
flux within 20 deg of the fibres; for comparison the same against the x-axis.

Prediction (group-velocity channelling, max quasi-P skew 65 deg for this CFRP):
flux in the optimized lens aligns with the converging fibres, not with the drive.

    PYTHONPATH=. .../dolfinx_complex/bin/python examples/analysis/lens_flux.py
"""
import os as _os
_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..", ".."))
import sys
import numpy as np
import ufl
from dolfinx import fem

sys.path.insert(0, (_ROOT + "/examples/dolfinx"))
import dolfinx_lens as DL

M_SP = 0.6


def flux_cells(ew):
    U = ew.U
    ur = ufl.as_vector([U[0], U[1]]); ui = ufl.as_vector([U[2], U[3]])
    D = ew._Dmat(ew.theta)
    sr = D * ew._evoigt(ur); si = D * ew._evoigt(ui)          # [sxx, syy, sxy]
    w = ew.omega
    # e^{+i w t} (the solver's convention): P = -1/2 Re(sigma conj(i w u))
    Px = 0.5 * w * (sr[0] * ui[0] + sr[2] * ui[1] - si[0] * ur[0] - si[2] * ur[1])
    Py = 0.5 * w * (sr[2] * ui[0] + sr[1] * ui[1] - si[2] * ur[0] - si[1] * ur[1])
    V = fem.functionspace(ew.domain, ("DG", 0, (2,)))
    f = fem.Function(V)
    f.interpolate(fem.Expression(ufl.as_vector([Px, Py]), V.element.interpolation_points()))
    return f.x.array.reshape(-1, 2)


def alignment(P, ang, mask):
    mag = np.linalg.norm(P, axis=1)
    phiP = np.arctan2(P[:, 1], P[:, 0])
    d = np.abs(np.angle(np.exp(2j * (phiP - ang)))) / 2          # in [0, pi/2]
    w = mag * mask
    return np.degrees(np.sum(w * d) / w.sum()), np.sum(w * (d < np.radians(20))) / w.sum()


def main():
    D = np.load("results/dolfinx_lens_data.npz")
    th = D["thopt"]
    out = {}
    for om in (35.0, 40.0, 45.0, 50.0):
        ew = DL.make(nx=130, omega=om)
        cent = DL.centroids(ew)
        mask = ((cent[:, 0] > 0.3) & (cent[:, 0] < DL.Lx - M_SP) &
                (cent[:, 1] > M_SP) & (cent[:, 1] < DL.Ly - M_SP)).astype(float)
        for name, t in (("straight", np.zeros(th.size)), ("lens", th)):
            ew.set_theta(t); ew.solve()
            P = flux_cells(ew)
            dfib, ffib = alignment(P, t, mask)
            dx, fx = alignment(P, np.zeros(th.size), mask)
            out[f"{name}_{int(om)}"] = (dfib, ffib, dx, fx)
            print(f"omega={om:4.0f} {name:8s}: flux-vs-fibre mean {dfib:5.1f} deg "
                  f"({100*ffib:4.1f}% within 20 deg) | flux-vs-x mean {dx:5.1f} deg ({100*fx:4.1f}%)")
            if om == 40.0:
                np.save(f"results/lens_flux_{name}.npy", np.column_stack([cent, P, t]))
    np.savez("results/lens_flux.npz", **{k: np.array(v) for k, v in out.items()})


if __name__ == "__main__":
    main()
