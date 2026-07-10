r"""
Conforming gmsh meshes for the dolfinx wave-control studies: a rectangular plate,
optionally with a circular hole whose boundary the mesh *conforms* to (a genuine
traction-free scatterer, not a soft SIMP density hole on a structured grid).
"""
from mpi4py import MPI
import gmsh
from dolfinx.io import gmshio


def rect_mesh(Lx, Ly, h, hole=None, comm=MPI.COMM_WORLD):
    """Triangulate [0,Lx]x[0,Ly] at element size h.  If hole=(cx,cy,r) is given,
    a circular disk is cut out so the mesh conforms to the hole boundary."""
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("plate")
    rect = gmsh.model.occ.addRectangle(0, 0, 0, Lx, Ly)
    if hole is not None:
        cx, cy, r = hole
        disk = gmsh.model.occ.addDisk(cx, cy, 0, r, r)
        gmsh.model.occ.cut([(2, rect)], [(2, disk)])
    gmsh.model.occ.synchronize()
    surfs = [s[1] for s in gmsh.model.getEntities(2)]
    gmsh.model.addPhysicalGroup(2, surfs, 1)
    # slightly finer near a hole for a clean scatterer boundary
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", h)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", h)
    gmsh.model.mesh.generate(2)
    mesh, _, _ = gmshio.model_to_mesh(gmsh.model, comm, 0, gdim=2)
    gmsh.finalize()
    return mesh
