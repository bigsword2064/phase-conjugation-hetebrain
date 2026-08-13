"""
Build an MRE storage-modulus field on a regular hexahedral box mesh.

Reads subject MRE stiffness/damping NIfTI volumes, re-orients them into the
mesh's world coordinates via the NIfTI affine, interpolates onto FEM dof
points, and writes the resulting storage-modulus field (adios4dolfinx + XDMF)
for later interpolation onto the head mesh (see mida_hete_deep_rev_first_fine.py).
"""

from pathlib import Path

import nibabel as nib
import numpy as np
import ufl
from dolfinx import fem
from dolfinx.io import XDMFFile
from dolfinx.mesh import CellType, create_box
from mpi4py import MPI
from scipy.interpolate import RegularGridInterpolator as rgi

import adios4dolfinx

SCRIPT_DIR = Path(__file__).resolve().parent

# --- Load NIfTI files ---
nii_stiffness = nib.load(SCRIPT_DIR / "MRE134_Stiffness3D.nii")
stiffness_data = nii_stiffness.get_fdata()
affine = nii_stiffness.affine  # voxel -> world

nii_damping = nib.load(SCRIPT_DIR / "MRE134_Damping3D.nii")
damping_data = nii_damping.get_fdata()

# --- Apply the scan-specific orientation ---
# Transpose (swap axes)
stiffness_data = np.transpose(stiffness_data, (1, 2, 0))
damping_data = np.transpose(damping_data, (1, 2, 0))

# Flip x-axis
stiffness_data = np.flip(stiffness_data, axis=0)
damping_data = np.flip(damping_data, axis=0)

Nx, Ny, Nz = stiffness_data.shape

# --- Build new world coordinate axes ---
# Mimic voxel indexing after the transpose+flip above
i_idx = np.arange(Nx)
j_idx = np.arange(Ny)
k_idx = np.arange(Nz)

# Undo the flip when mapping back to original voxel indices
i_vox = Nx - 1 - i_idx
j_vox = j_idx
k_vox = k_idx


def voxel_to_world(ii, jj, kk):
    vox = np.vstack([ii, jj, kk, np.ones_like(ii)])
    return (affine @ vox).T[:, :3]


# Manual offsets below align the scan's world coordinates with the head mesh
# (fit against the MIDA atlas registration for this subject).
# X-axis: vary i_vox
x_world = voxel_to_world(i_vox, np.zeros_like(i_vox), np.zeros_like(i_vox))[:, 0] + 130.9167 + 10 - 3
# Y-axis: vary j_vox
y_world = voxel_to_world(np.zeros_like(j_vox), j_vox, np.zeros_like(j_vox))[:, 1] + 48 + 68.1222 + 3
# Z-axis: vary k_vox
z_world = voxel_to_world(np.zeros_like(k_vox), np.zeros_like(k_vox), k_vox)[:, 2] - 32 + 15.2289 - 3.2289 + 4

# --- Interpolators ---
interp_stiffness = rgi((x_world, y_world, z_world), stiffness_data, bounds_error=False, fill_value=0)
interp_damping = rgi((x_world, y_world, z_world), damping_data, bounds_error=False, fill_value=0)

# --- Mesh over bounding box ---
xmin, xmax = x_world.min(), x_world.max()
ymin, ymax = y_world.min(), y_world.max()
zmin, zmax = z_world.min(), z_world.max()

domain = create_box(
    MPI.COMM_WORLD,
    [np.array([xmin, ymin, zmin]), np.array([xmax, ymax, zmax])],
    [Nx - 1, Ny - 1, Nz - 1],
    CellType.hexahedron,
)

# --- FEM setup ---
Q_element = fem.functionspace(domain, ("DG", 0))
point_scalar_space = fem.functionspace(domain, ("Lagrange", 1))

stiffness_scalar_function = fem.Function(point_scalar_space)
damping_scalar_function = fem.Function(point_scalar_space)

Q_dof_coords = point_scalar_space.tabulate_dof_coordinates()

# Interpolate values onto each dof point (RegularGridInterpolator returns a
# length-1 array per point, so index [0] to get the scalar).
for dof, coord in enumerate(Q_dof_coords):
    stiffness_scalar_function.x.array[dof] = interp_stiffness(coord)[0]
    damping_scalar_function.x.array[dof] = interp_damping(coord)[0]

# Compute storage modulus
storage_function = (
    stiffness_scalar_function
    * (1 + ufl.sqrt(1 + 4 * damping_scalar_function**2))
    / (2 * (1 + 4 * damping_scalar_function**2))
)

E_expr = fem.Expression(storage_function, Q_element.element.interpolation_points)
stiffnessh = fem.Function(Q_element, name="stiffness")
stiffnessh.interpolate(E_expr)

# --- Save results ---
stiffness_file = SCRIPT_DIR / "storage_field_affine_rotated.bp"
adios4dolfinx.write_mesh(stiffness_file, domain)
adios4dolfinx.write_function(stiffness_file, stiffnessh, time=0.0, name="stiffness")

with XDMFFile(domain.comm, str(SCRIPT_DIR / "cube_storagefield_affine_rotated.xdmf"), "w") as xdmf:
    xdmf.write_mesh(domain)
    xdmf.write_function(stiffnessh, 0)