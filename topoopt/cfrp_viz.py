"""
Visualization of CFRP toolpath-integrated designs.

Renders the fiber layout the way the paper does (Wong et al. 2026, Fig. 6):
  * structural density (composite topology),
  * fiber orientation streamlines -> directional anisotropy of the fiber field,
  * the fiber/matrix material state chi_hat -> the actual towpregs ("toolpaths"),
  * the |curl| contour map -> where fiber curvature concentrates.
"""
from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _grid(arr, mesh):
    return np.asarray(arr).reshape(mesh.nely, mesh.nelx)


def clean_density(z, mesh, R, beta=8.0, eta=0.5):
    """Physical density for display: linear filter (eq 2) + Heaviside project.

    The optimizer's raw design variable z carries sub-filter-radius speckle;
    the *physical* field the FE analysis sees is the filtered density, and a
    mild Heaviside projection crisps it to a clean, hole-free topology
    (zero isolated pixels / interior holes for the converged designs).
    """
    from .cfrp_problem import density_filter
    from .cfrp import heaviside
    P = density_filter(mesh, R)
    y = P @ np.asarray(z)
    return heaviside(y, eta, beta)


def orientation_streamlines(ax, mesh, theta, density, *, dens_thresh=0.4,
                            color=None, density_bg=True, seed_density=2.5,
                            linewidth=0.7):
    """Streamlines of the unit fiber field v=(cos t, sin t) over the domain.

    Fibers are headless lines (theta and theta+pi are the same fiber), so we
    draw the field as streamlines masked to the structural (solid) region.
    """
    tg = _grid(theta, mesh)
    dg = _grid(density, mesh)
    u = np.cos(tg)
    v = np.sin(tg)
    mask = dg < dens_thresh
    u = np.ma.array(u, mask=mask)
    v = np.ma.array(v, mask=mask)

    xs = (np.arange(mesh.nelx) + 0.5) * mesh.dx
    ys = (np.arange(mesh.nely) + 0.5) * mesh.dy

    if density_bg:
        ax.imshow(dg, origin="lower", cmap="Greys", vmin=0, vmax=1.6,
                  extent=[0, mesh.Lx, 0, mesh.Ly], aspect="equal", alpha=0.85)
    if color is None:
        color = "tab:red"
    ax.streamplot(xs, ys, u, v, density=seed_density, color=color,
                  linewidth=linewidth, arrowsize=1e-6)
    ax.set_xlim(0, mesh.Lx); ax.set_ylim(0, mesh.Ly)
    ax.set_aspect("equal")


def fiber_towpregs(ax, mesh, chi_hat, density, *, dens_thresh=0.4):
    """Render the fiber(1)/matrix(0) state as towpregs inside the structure."""
    cg = _grid(chi_hat, mesh)
    dg = _grid(density, mesh)
    void = dg < dens_thresh
    # solid+matrix = light orange, solid+fiber = black, void = white
    img = np.ones(cg.shape)            # 1 -> light
    img = np.where(cg > 0.5, 0.0, 0.85)  # fiber dark, matrix light-orange-ish
    img = np.ma.array(img, mask=void)
    ax.imshow(img, origin="lower", cmap="copper_r", vmin=0, vmax=1,
              extent=[0, mesh.Lx, 0, mesh.Ly], aspect="equal")
    ax.set_aspect("equal")


def curl_map(ax, mesh, curl, density, *, dens_thresh=0.3, vmax=None):
    cg = np.abs(_grid(curl, mesh))
    dg = _grid(density, mesh)
    cg = np.ma.array(cg, mask=(dg < dens_thresh))
    im = ax.imshow(cg, origin="lower", cmap="inferno", vmax=vmax,
                   extent=[0, mesh.Lx, 0, mesh.Ly], aspect="equal")
    return im


def plot_design(mesh, res, path, title="", R=None):
    """Four-panel summary figure for one optimized CFRP design.

    If R (filter radius) is given, the *physical* (filtered + projected)
    density is shown so the topology is clean and continuous; otherwise the
    raw design variable z is shown.
    """
    z = res["z"]; theta = res["theta"]; chi = res["chi_hat"]; curl = res["curl"]
    disp = clean_density(z, mesh, R) if R is not None else np.asarray(z)
    fig, ax = plt.subplots(2, 2, figsize=(14, 8.5))

    dg = _grid(disp, mesh)
    im0 = ax[0, 0].imshow(dg, origin="lower", cmap="gray_r", vmin=0, vmax=1,
                          extent=[0, mesh.Lx, 0, mesh.Ly], aspect="equal")
    ax[0, 0].set_title(f"composite density   f = {float(res['f']):.3f}")
    plt.colorbar(im0, ax=ax[0, 0], fraction=0.035, pad=0.03)

    orientation_streamlines(ax[0, 1], mesh, theta, disp)
    ax[0, 1].set_title("fiber orientation streamlines (directional anisotropy)")

    fiber_towpregs(ax[1, 0], mesh, chi, disp)
    ax[1, 0].set_title("fiber / matrix towpregs  (toolpaths)")

    im3 = curl_map(ax[1, 1], mesh, curl, disp)
    ax[1, 1].set_title(f"|curl|   max = {float(res['curl_max']):.2f} /m")
    plt.colorbar(im3, ax=ax[1, 1], fraction=0.035, pad=0.03)

    for a in ax.ravel():
        a.set_xlabel("x [m]"); a.set_ylabel("y [m]")
    fig.suptitle(title, y=1.0, fontsize=13)
    plt.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print("saved", path)
