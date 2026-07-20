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


def _holelist(hole):
    """Normalize `hole` (None | single (cx,cy,r) | list of them) to a list."""
    if hole is None:
        return []
    if isinstance(hole[0], (int, float, np.floating, np.integer)):
        return [hole]
    return list(hole)


def _director_grid(cents, theta, xlim, ylim, n=240, hole=None):
    """Regular grid of the double-angle director components (C,S)=(cos2t,sin2t),
    with points inside any hole=(cx,cy,r) (or list of holes) set to NaN."""
    xs = np.linspace(xlim[0], xlim[1], n)
    ys = np.linspace(ylim[0], ylim[1], int(n*(ylim[1]-ylim[0])/(xlim[1]-xlim[0])))
    gx, gy = np.meshgrid(xs, ys)
    C = griddata(cents, np.cos(2*theta), (gx, gy), method="linear")
    S = griddata(cents, np.sin(2*theta), (gx, gy), method="linear")
    Cn = griddata(cents, np.cos(2*theta), (gx, gy), method="nearest")
    Sn = griddata(cents, np.sin(2*theta), (gx, gy), method="nearest")
    C = np.where(np.isnan(C), Cn, C); S = np.where(np.isnan(S), Sn, S)
    for cx, cy, r in _holelist(hole):
        m = np.hypot(gx-cx, gy-cy) < r*1.02
        C[m] = np.nan; S[m] = np.nan
    return xs, ys, C, S


def plot_director_field(ax, cents, theta, xlim, ylim, holes=None, n=26,
                        cmap="hsv", lw=1.7, length=None):
    r"""Vector (director) map of the anisotropic fiber orientation: at a regular
    grid of points, a short line segment aligned with the local fiber direction
    $(\cos\theta,\sin\theta)$, coloured by the orientation angle through a cyclic
    colormap (period $\pi$, since a fiber is a director).  This is the raw
    anisotropy field the optimizer designs, before toolpath delineation.  Segments
    inside a hole are omitted.  Returns the LineCollection (for a colorbar)."""
    from matplotlib.collections import LineCollection
    xs = np.linspace(xlim[0], xlim[1], n)
    ys = np.linspace(ylim[0], ylim[1], int(n*(ylim[1]-ylim[0])/(xlim[1]-xlim[0])))
    gx, gy = np.meshgrid(xs, ys)
    C = griddata(cents, np.cos(2*theta), (gx, gy), method="linear")
    S = griddata(cents, np.sin(2*theta), (gx, gy), method="linear")
    Cn = griddata(cents, np.cos(2*theta), (gx, gy), method="nearest")
    Sn = griddata(cents, np.sin(2*theta), (gx, gy), method="nearest")
    C = np.where(np.isnan(C), Cn, C); S = np.where(np.isnan(S), Sn, S)
    ang = 0.5*np.arctan2(S, C)
    if length is None:
        length = 0.42*(xs[1]-xs[0])
    dxx = length*np.cos(ang); dyy = length*np.sin(ang)
    segs, cols = [], []
    for i in range(gx.shape[0]):
        for j in range(gx.shape[1]):
            px, py = gx[i, j], gy[i, j]
            if any(np.hypot(px-cx, py-cy) < r for cx, cy, r in _holelist(holes)):
                continue
            segs.append([(px-dxx[i, j], py-dyy[i, j]),
                         (px+dxx[i, j], py+dyy[i, j])])
            cols.append(ang[i, j] % np.pi)
    lc = LineCollection(segs, array=np.array(cols), cmap=cmap, linewidths=lw)
    lc.set_clim(0, np.pi)
    ax.add_collection(lc)
    th = np.linspace(0, 2*np.pi, 80)
    for cx, cy, r in _holelist(holes):
        ax.fill(cx+r*np.cos(th), cy+r*np.sin(th), color="white", ec="0.4",
                lw=1.0, zorder=5)
    ax.set_xlim(xlim); ax.set_ylim(ylim); ax.set_aspect("equal")
    return lc


def _orient_consistent(fx, fy, V, ref=(1.0, 0.0)):
    """Region-grow a globally sign-consistent director branch: BFS over the valid
    grid, flipping each pixel to align with an already-visited neighbour.  This
    removes the hard reference seam that otherwise contaminates the phase solve.
    The seed is aligned to `ref` so the branch has a definite global sense."""
    from collections import deque
    Ny, Nx = V.shape
    ofx = fx.copy(); ofy = fy.copy()
    done = np.zeros((Ny, Nx), bool)
    seeds = np.argwhere(V)
    if len(seeds) == 0:
        return ofx, ofy
    sy, sx = seeds[len(seeds)//2]
    if ofx[sy, sx]*ref[0] + ofy[sy, sx]*ref[1] < 0:
        ofx[sy, sx] *= -1; ofy[sy, sx] *= -1
    done[sy, sx] = True; q = deque([(sy, sx)])
    while q:
        iy, ix = q.popleft()
        for jy, jx in ((iy, ix+1), (iy, ix-1), (iy+1, ix), (iy-1, ix)):
            if 0 <= jy < Ny and 0 <= jx < Nx and V[jy, jx] and not done[jy, jx]:
                if ofx[jy, jx]*ofx[iy, ix] + ofy[jy, jx]*ofy[iy, ix] < 0:
                    ofx[jy, jx] *= -1; ofy[jy, jx] *= -1
                done[jy, jx] = True; q.append((jy, jx))
    return ofx, ofy


def fiber_phase(cents, theta, xlim, ylim, n=240, holes=None, ref=(1.0, 0.0)):
    r"""Phase-field (level-set) delineation of the fiber toolpaths, following the
    reference-paper "wave projection".  The tows are the evenly spaced level sets
    of a scalar phase $\psi$ whose gradient is unit and perpendicular to the fibers
    ($\nabla\psi=[-\sin\theta,\cos\theta]$), so $\psi$'s contours run *along* the
    fibers and are spaced by a constant tow pitch.  We orient the (sign-ambiguous)
    director to a smooth branch, then match the gradient directly by least squares
    $\min\|\nabla\psi-\mathbf g\|^2$ (so $|\nabla\psi|\approx1$, i.e. constant
    pitch; holes masked out).  Returns (xs, ys, psi), NaN in the voids/outside."""
    import scipy.sparse as sp
    from scipy.sparse.linalg import lsqr
    xs, ys, C, S = _director_grid(cents, theta, xlim, ylim, n=n, hole=holes)
    Ny, Nx = C.shape
    dx = xs[1]-xs[0]; dy = ys[1]-ys[0]
    V = np.isfinite(C) & np.isfinite(S)
    A = 0.5*np.arctan2(np.where(V, S, 0.0), np.where(V, C, 1.0))    # director angle
    fx = np.cos(A); fy = np.sin(A)
    fx, fy = _orient_consistent(fx, fy, V, ref)    # smooth (seam-free) branch
    gx = -fy; gy = fx                              # grad(psi): unit, perp to fiber
    idx = -np.ones((Ny, Nx), int); idx[V] = np.arange(int(V.sum()))
    N = int(V.sum())
    # match the gradient directly (forward differences on interior valid edges)
    rows, cols, vals, rhs = [], [], [], []; eq = 0
    for iy, ix in np.argwhere(V):
        p = idx[iy, ix]
        if ix+1 < Nx and V[iy, ix+1]:
            rows += [eq, eq]; cols += [idx[iy, ix+1], p]
            vals += [1.0/dx, -1.0/dx]; rhs.append(gx[iy, ix]); eq += 1
        if iy+1 < Ny and V[iy+1, ix]:
            rows += [eq, eq]; cols += [idx[iy+1, ix], p]
            vals += [1.0/dy, -1.0/dy]; rhs.append(gy[iy, ix]); eq += 1
    seeds = np.argwhere(V); sy, sx = seeds[len(seeds)//2]
    rows += [eq]; cols += [idx[sy, sx]]; vals += [1.0]; rhs.append(0.0); eq += 1
    Amat = sp.csr_matrix((vals, (rows, cols)), shape=(eq, N))
    sol = lsqr(Amat, np.asarray(rhs), atol=1e-9, btol=1e-9, iter_lim=6000)[0]
    psi = np.full((Ny, Nx), np.nan); psi[V] = sol
    return xs, ys, psi


def plot_toolpaths_phase(ax, cents, theta, xlim, ylim, holes=None, spacing=0.16,
                         n=340, ref=(1.0, 0.0), cmap="gray_r", mode="stripe",
                         color="k", lw=0.8, dens=None, dens_thresh=0.5):
    r"""Draw clean, evenly spaced fiber toolpaths from the fiber phase
    (:func:`fiber_phase`).  ``mode="stripe"`` renders the reference-paper wave
    projection $\chi=\tfrac12+\tfrac12\cos(2\pi\psi/d)$ as a woven tow/gap field
    (black tows, constant pitch $d$=``spacing``); ``mode="line"`` draws the tow
    centre-lines as evenly spaced level sets.  Tows never enter the holes (masked
    in the phase solve).  If ``dens`` (a per-``cents`` density) is given, tows are
    shown only where the interpolated density exceeds ``dens_thresh`` -- i.e.\ only
    where there is solid fiber material (used for the metamaterial unit cell)."""
    from scipy.interpolate import griddata as _gd
    xs, ys, psi = fiber_phase(cents, theta, xlim, ylim, n=n, holes=holes, ref=ref)
    solidmask = None
    if dens is not None:
        gx, gy = np.meshgrid(xs, ys)
        dg = _gd(cents, np.asarray(dens), (gx, gy), method="linear")
        dgn = _gd(cents, np.asarray(dens), (gx, gy), method="nearest")
        dg = np.where(np.isnan(dg), dgn, dg)
        solidmask = dg >= dens_thresh
    if mode == "stripe":
        chi = 0.5 + 0.5*np.cos(2*np.pi*psi/spacing)
        if solidmask is not None:
            chi = np.where(solidmask, chi, np.nan)     # blank the matrix/void
        ax.imshow(chi, origin="lower", extent=(xlim[0], xlim[1], ylim[0], ylim[1]),
                  cmap=cmap, interpolation="bilinear", vmin=0, vmax=1, zorder=0)
    else:
        lo, hi = np.nanmin(psi), np.nanmax(psi)
        ax.contour(xs, ys, psi, levels=np.arange(lo, hi + spacing, spacing),
                   colors=color, linewidths=lw)
    th = np.linspace(0, 2*np.pi, 80)
    for cx, cy, r in _holelist(holes):
        ax.fill(cx+r*np.cos(th), cy+r*np.sin(th), color="white", ec="0.4",
                lw=1.0, zorder=5)
    ax.set_xlim(xlim); ax.set_ylim(ylim); ax.set_aspect("equal")


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
            if any(np.hypot(p[0]-cx, p[1]-cy) < r*1.02
                   for cx, cy, r in _holelist(hole)):
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
    holes = _holelist(hole)
    for yy in sy:
        for xx in sx:
            if any(np.hypot(xx-cx, yy-cy) < r*1.15 for cx, cy, r in holes):
                continue
            line = _integrate(Ci, Si, (xx, yy), xlim, ylim, hole, ds=ds)
            if len(line) > 4:
                ax.plot(line[:, 0], line[:, 1], color=color, lw=lw, alpha=alpha)
    th = np.linspace(0, 2*np.pi, 80)
    for cx, cy, r in holes:
        ax.fill(cx+r*np.cos(th), cy+r*np.sin(th),
                color="white", ec="0.3", lw=1.2, zorder=5)
    ax.set_xlim(xlim); ax.set_ylim(ylim); ax.set_aspect("equal")
