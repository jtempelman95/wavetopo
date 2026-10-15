r"""Is the orientation lens REDIRECTING energy or AMPLIFYING the field?  And is it a
broadband (graded-index refraction) or a narrowband (interference/resonance) device?

For the saved 78x lens design (results/dolfinx_lens_data.npz, omega = 40) and the
straight-fibre baseline, at each drive frequency omega in a sweep:

  E_f     focus-region energy  int_reg |u|^2
  E_int   interior energy (outside the absorbing sponge)  int_int |u|^2
  P_in    time-averaged power injected by the force source,
          P_in = (omega/2) |int f . Im(u)|   (e^{+i omega t}, real force)
  phi     focus fraction E_f / E_int  (redistribution)
  G       = E_f / E_f0                (the paper's 'focus-energy gain')
  G_phi   = phi / phi0                (redirection alone)
  G_P     = (E_f/P_in) / (E_f0/P_in0) (focus energy per unit input power)

so G = G_phi * (E_int/E_int0): the gain splits exactly into redirection and
total-field growth, and P_in says how much of the growth is extra power drawn
from the (force-controlled) source rather than stored energy.

    .../dolfinx_complex/bin/python examples/analysis/lens_mechanism.py
"""
import os as _os
_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..", ".."))
import sys
import numpy as np
import ufl
from dolfinx import fem

sys.path.insert(0, (_ROOT + "/examples/dolfinx"))
import dolfinx_lens as DL
from wavetopo.dolfinx_elastic import _SUM

M_SP = 0.6


def metrics(ew):
    U = ew.U
    u2 = sum(U[i] * U[i] for i in range(4))
    x = ufl.SpatialCoordinate(ew.domain)
    interior = ufl.conditional(ufl.And(ufl.And(ufl.lt(x[0], DL.Lx - M_SP), ufl.gt(x[1], M_SP)),
                                       ufl.lt(x[1], DL.Ly - M_SP)), 1.0, 0.0)
    comm = ew.domain.comm
    Ef = ew.focus_energy()
    Ei = comm.allreduce(fem.assemble_scalar(fem.form(interior * u2 * ufl.dx)), op=_SUM())
    # f = (fr_x, fr_y, 0, 0); Im u = (U2, U3)
    fu = comm.allreduce(fem.assemble_scalar(fem.form((ew.fr[0] * U[2] + ew.fr[1] * U[3]) * ufl.dx)), op=_SUM())
    return Ef, Ei, 0.5 * ew.omega * abs(fu)


def main():
    d = np.load("results/dolfinx_lens_data.npz")
    th = d["thopt"]
    omegas = np.arange(20.0, 62.5, 2.5)
    rows = []
    for om in omegas:
        ew = DL.make(nx=130, omega=om)
        ew.set_theta(np.zeros(th.size)); ew.solve(); E0, I0, P0 = metrics(ew)
        ew.set_theta(th); ew.solve(); E1, I1, P1 = metrics(ew)
        G, Gphi, GP = E1 / E0, (E1 / I1) / (E0 / I0), (E1 / P1) / (E0 / P0)
        rows.append((om, E0, I0, P0, E1, I1, P1, G, Gphi, GP))
        print(f"omega={om:5.1f}  G={G:7.1f}  G_phi(redirect)={Gphi:5.2f}  interior growth={I1/I0:6.2f}  "
              f"P_in ratio={P1/P0:5.2f}  G_P(per power)={GP:6.1f}  phi0={100*E0/I0:.2f}% phi1={100*E1/I1:.2f}%")
    np.savez("results/lens_mechanism.npz", rows=np.array(rows),
             cols="omega,E0,I0,P0,E1,I1,P1,G,Gphi,GP")
    print("wrote results/lens_mechanism.npz")


if __name__ == "__main__":
    main()
