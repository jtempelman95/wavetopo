"""CFRP lamina stiffness helpers used by the analysis scripts (self-contained copy of the two
functions shared with the companion NLTopt code, so this repository runs on its own)."""
import numpy as np

E1, E2, G12, NU12 = 131e9, 9e9, 5e9, 0.27


def lamina_Q(E1=E1, E2=E2, G12=G12, nu12=NU12):
    """On-axis plane-stress reduced stiffness matrix Q (3x3, Voigt 1,2,6)."""
    nu21 = nu12 * E2 / E1
    d = 1.0 - nu12 * nu21
    return np.array([[E1 / d, nu12 * E2 / d, 0.0], [nu12 * E2 / d, E2 / d, 0.0], [0.0, 0.0, G12]])


def Q_family(s, Q=None):
    """Anisotropy family Q(s) = Q_iso + s (Q - Q_iso); Q_iso shares Q's invariants U1, U5
    (U4 = U1 - 2 U5). s = 0 is isotropic, s = 1 the CFRP lamina; U2, U3 scale exactly as s."""
    Q = lamina_Q() if Q is None else Q
    Q11, Q12, Q22, Q66 = Q[0, 0], Q[0, 1], Q[1, 1], Q[2, 2]
    U1 = (3 * Q11 + 3 * Q22 + 2 * Q12 + 4 * Q66) / 8.0
    U5 = (Q11 + Q22 - 2 * Q12 + 4 * Q66) / 8.0
    U4 = U1 - 2.0 * U5
    Qiso = np.array([[U1, U4, 0.0], [U4, U1, 0.0], [0.0, 0.0, U5]])
    return Qiso + s * (Q - Qiso)
