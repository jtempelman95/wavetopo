r"""
Conforming gmsh meshes for the dolfinx wave-control studies: a rectangular plate,
optionally with one or more circular holes whose boundaries the mesh *conforms*
to -- genuine traction-free scatterers, not soft SIMP density holes staircased
onto a structured grid.

The distinction matters for the through-hole studies.  A soft-void hole on a
structured mesh replaces the circle by the set of elements whose centroid falls
inside it, so the boundary is a staircase: it scatters off its own corners, the
rim concentrates spurious energy, and the "hole radius" the design actually sees
is only accurate to one element.  Cutting the disk in the geometry instead gives
an exactly circular, traction-free boundary (the natural BC of the weak form) and
lets the element size be refined where the scattering happens.
"""
from mpi4py import MPI
import numpy as np
import gmsh
from dolfinx.io import gmshio


def rect_mesh(Lx, Ly, h, hole=None, holes=None, h_hole=None, refine_dist=2.0,
              comm=MPI.COMM_WORLD):
    """Triangulate [0,Lx]x[0,Ly] at element size h, conforming to circular holes.

    Parameters
    ----------
    hole : (cx, cy, r) or None
        Single hole (kept for backward compatibility).
    holes : list of (cx, cy, r) or None
        Any number of holes; takes precedence over ``hole``.
    h_hole : float or None
        Element size ON the hole boundaries.  Defaults to ``h/2`` so the circular
        scatterer is resolved more finely than the bulk.  The size grades smoothly
        back to ``h`` over ``refine_dist`` hole radii.
    refine_dist : float
        Distance (in units of the smallest hole radius) over which the refinement
        relaxes back to the bulk size.
    """
    if holes is None:
        holes = [hole] if hole is not None else []
    holes = [tuple(map(float, hxy)) for hxy in holes]

    if gmsh.isInitialized():
        gmsh.finalize()
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("plate")

    rect = gmsh.model.occ.addRectangle(0, 0, 0, Lx, Ly)
    if holes:
        disks = [(2, gmsh.model.occ.addDisk(cx, cy, 0, r, r))
                 for cx, cy, r in holes]
        gmsh.model.occ.cut([(2, rect)], disks)
    gmsh.model.occ.synchronize()

    surfs = [s[1] for s in gmsh.model.getEntities(2)]
    gmsh.model.addPhysicalGroup(2, surfs, 1)

    gmsh.option.setNumber("Mesh.MeshSizeMax", h)
    gmsh.option.setNumber("Mesh.MeshSizeMin", h)

    if holes:
        # identify the arcs bounding each hole: their centre of mass sits at the
        # hole centre (the outer rectangle edges do not)
        rmin = min(r for _, _, r in holes)
        hh = h_hole if h_hole is not None else 0.5*h
        hole_curves = []
        for dim, tag in gmsh.model.getEntities(1):
            cm = np.array(gmsh.model.occ.getCenterOfMass(dim, tag))
            for cx, cy, r in holes:
                if np.hypot(cm[0]-cx, cm[1]-cy) < 0.5*r:
                    hole_curves.append(tag)
                    break
        if hole_curves:
            fd = gmsh.model.mesh.field.add("Distance")
            gmsh.model.mesh.field.setNumbers(fd, "CurvesList", hole_curves)
            gmsh.model.mesh.field.setNumber(fd, "Sampling", 200)
            ft = gmsh.model.mesh.field.add("Threshold")
            gmsh.model.mesh.field.setNumber(ft, "InField", fd)
            gmsh.model.mesh.field.setNumber(ft, "SizeMin", hh)
            gmsh.model.mesh.field.setNumber(ft, "SizeMax", h)
            gmsh.model.mesh.field.setNumber(ft, "DistMin", 0.25*rmin)
            gmsh.model.mesh.field.setNumber(ft, "DistMax", refine_dist*rmin)
            gmsh.model.mesh.field.setAsBackgroundMesh(ft)
            # let the field alone drive the size
            gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
            gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
            gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
            gmsh.option.setNumber("Mesh.MeshSizeMin", min(h, hh))

    gmsh.model.mesh.generate(2)
    mesh, _, _ = gmshio.model_to_mesh(gmsh.model, comm, 0, gdim=2)
    gmsh.finalize()
    return mesh
