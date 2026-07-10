r"""
Toolpath (fiber-orientation) visualization for the dolfinx wave-control designs.

The orientation is a per-cell (DG0) director angle theta.  A fiber is a line, so
theta and theta+pi are physically identical; naive streamlines of (cos t, sin t)
break at the +-pi/2 wrap.  We therefore interpolate the *double-angle* (nematic)
field (cos 2t, sin 2t) -- which is single-valued and free of the sign ambiguity
-- onto a regular grid, recover the local director, and integrate streamlines by
hand while enforcing sign-continuity (each step is aligned to the previous one).
The result is a set of smooth, physically meaningful fiber toolpaths.
"""
import numpy as np
from scipy.interpolate import griddata, RegularGridInterpolator


def _director_grid(cents, theta, xlim, ylim, n=240, hole=None):
    """Regular grid of the double-angle director components (C,S)=(cos2t,sin2t),
    with points inside an optional hole=(cx,cy,r) set to NaN."""
    xs = np.linspace(xlim[0], xlim[1], n)
    ys = np.linspace(ylim[0], ylim[1], int(n*(ylim[1]-ylim[0])/(xlim[1]-xlim[0])))
    gx, gy = np.meshgrid(xs, ys)
    C = griddata(cents, np.cos(2*theta), (gx, gy), method="linear")
    S = griddata(cents, np.sin(2*theta), (gx, gy), method="linear")
    Cn = griddata(cents, np.cos(2*theta), (gx, gy), method="nearest")
    Sn = griddata(cents, np.sin(2*theta), (gx, gy), method="nearest")
    C = np.where(np.isnan(C), Cn, C); S = np.where(np.isnan(S), Sn, S)
    if hole is not None:
        cx, cy, r = hole
        m = np.hypot(gx-cx, gy-cy) < r*1.02
        C[m] = np.nan; S[m] = np.nan
    return xs, ys, C, S


def _integrate(iC, iS, seed, xlim, ylim, hole, ds=0.02, nmax=1200):
    """Integrate one fiber line through the director field from a seed, both
    directions, choosing the director sign for continuity."""
    def director(p):
        c = iC(p); s = iS(p)
        if not np.isfinite(c) or not np.isfinite(s):
            return None
        a = 0.5*np.arctan2(s, c)
        return np.array([np.cos(a), np.sin(a)])

    def march(sign):
        p = np.array(seed, float); prev = None; pts = [p.copy()]
        for _ in range(nmax):
            d = director(p)
            if d is None:
                break
            if prev is not None and np.dot(d, prev) < 0:
                d = -d
            elif prev is None:
                d = sign*d
            prev = d
            p = p + ds*d
            if not (xlim[0] <= p[0] <= xlim[1] and ylim[0] <= p[1] <= ylim[1]):
                break
            if hole is not None and np.hypot(p[0]-hole[0], p[1]-hole[1]) < hole[2]*1.02:
                break
            pts.append(p.copy())
        return pts

    fwd = march(+1.0); bwd = march(-1.0)
    return np.array(bwd[::-1] + fwd[1:])


def plot_toolpaths(ax, cents, theta, xlim, ylim, hole=None, nseed=26,
                   color="k", lw=0.9, alpha=0.85, ds=0.02):
    """Draw fiber toolpaths (director streamlines) on ax."""
    xs, ys, C, S = _director_grid(cents, theta, xlim, ylim, hole=hole)
    iC = RegularGridInterpolator((ys, xs), C, bounds_error=False, fill_value=np.nan)
    iS = RegularGridInterpolator((ys, xs), S, bounds_error=False, fill_value=np.nan)
    Ci = lambda p: iC((p[1], p[0])); Si = lambda p: iS((p[1], p[0]))
    # seeds on a coarse grid, skipping the hole
    sx = np.linspace(xlim[0]+0.05, xlim[1]-0.05, nseed)
    sy = np.linspace(ylim[0]+0.05, ylim[1]-0.05, max(6, int(nseed*(ylim[1]-ylim[0])/(xlim[1]-xlim[0]))))
    for yy in sy:
        for xx in sx:
            if hole is not None and np.hypot(xx-hole[0], yy-hole[1]) < hole[2]*1.15:
                continue
            line = _integrate(Ci, Si, (xx, yy), xlim, ylim, hole, ds=ds)
            if len(line) > 4:
                ax.plot(line[:, 0], line[:, 1], color=color, lw=lw, alpha=alpha)
    if hole is not None:
        th = np.linspace(0, 2*np.pi, 80)
        ax.fill(hole[0]+hole[2]*np.cos(th), hole[1]+hole[2]*np.sin(th),
                color="white", ec="0.3", lw=1.2, zorder=5)
    ax.set_xlim(xlim); ax.set_ylim(ylim); ax.set_aspect("equal")
