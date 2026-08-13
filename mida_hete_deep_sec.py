"""
Phase-conjugation (time-reversal) MRE simulation on the heterogeneous MIDA
head mesh.

Rebuilds the same material fields as mida_hete_deep_first.py, then
replaces the forward Gaussian source with a phase-conjugated reaction load
built from that script's saved solution (brain_uh_mpi_*.npy,
brain_ph_mpi_*.npy) on the loading boundary. Must be run after
mida_hete_deep_first.py has produced those files.
"""

import gc
import os
import time
from pathlib import Path

import numpy as np
import psutil
import ufl
from basix.ufl import element, mixed_element
from mpi4py import MPI

from dolfinx import default_scalar_type, fem, io, mesh
from dolfinx.fem import (
    Function,
    create_interpolation_data,
    dirichletbc,
    functionspace,
    locate_dofs_topological,
)
from dolfinx.fem.petsc import LinearProblem
from dolfinx.io import gmsh as gmshio
from dolfinx.mesh import locate_entities_boundary
from ufl import TestFunctions, TrialFunctions, dx, grad


def get_memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 ** 2  # Memory usage in MB
# Start monitoring memory usage
start_memory = get_memory_usage()

start_time = time.time()

# unit: mm, kg, s
# read mesh
SCRIPT_DIR = Path(__file__).resolve().parent
filepath = str(SCRIPT_DIR / "input" / "MIDA_processed_half_group_0.96.msh")
x0, y0, z0 = 121, 56, 24  # source location (already positioned at the anisotropy site)

meshdata = gmshio.read_from_msh(filepath, MPI.COMM_WORLD, gdim=3)
domain = meshdata.mesh
cell_tags = meshdata.cell_tags
if cell_tags is not None:
    unique_tags = np.unique(cell_tags.values)
    print("Unique cell tags:", unique_tags)
else:
    print("No cell tags found.")

num_cells = domain.topology.index_map(domain.topology.dim).size_local
print(f"Number of 3D elements (cells): {num_cells}")

kap_wm = 2.19e6  # from tube test
kap_gm = 2.19e6  # from tube test
kap_csf = 2.19e6  # from  paper, edlucidating viscoelastic effects on FU
kap_b = 4.76e6  # from amir paper
kap_s = 3.36e6  # from amir paper
kap_m =  2.3e6 # from Armen Elastic properties of soft tissue

# G_wm = 1.4e5*(1+1j) # from tube test
# G_gm = 1.4e5*(1+1j)# from tube test
G_wm = 61.7238+2.2815*1j # from prony-series calculation, corresponding to 350kHz
G_gm = 61.7238+2.2815*1j # from prony-series calculation, corresponding to 350kHz
G_csf = 22  # from  paper, edlucidating viscoelastic effects on FU
G_b = 3.28e6  # from amir paper
G_s = 6.7e2  # from amir paper
G_m = 110/3  # from Armen Elastic properties of soft tissue

rho_wm = 1060e-9  # from amir paper
rho_gm = 1060e-9  # from amir paper
rho_csf = 1040e-9  # from  paper, edlucidating viscoelastic effects on FU
rho_b = 1721e-9  # from amir paper
rho_s = 1100e-9  # from amir paper
rho_m = 1060e-9  # Samuel, density and hyration of fresh and fixed human skeletal muscle

# define the load magnitude and frequency
p0 = 300.0 #300kpa
f0 = 350e3 #Hz
omega = 2.0 * np.pi * f0


# functions to define subdomain for fixed faces and loaded faces
def absorbing_face(x):
    return np.logical_or(x[1] <= 1.0, x[2] <= 1.0)

def fix_face(x):
    return np.logical_and(np.logical_and(x[1]>=75.0,x[1]<=85.0), x[2] > 1.0)

def load_face(x):
    return np.logical_and(x[1]>90.0,x[2]>1.0)

# define functionspace and trial,test function, define specially for the mixed formulations
ue = element("Lagrange", domain.basix_cell(), 2,shape=(domain.topology.dim,))
pe = element("Lagrange", domain.basix_cell(), 1)

V_el = mixed_element([ue, pe])
V = fem.functionspace(domain, V_el)
dofmap = V.dofmap
index_map = dofmap.index_map
# Get the total number of DOFs (global)
num_dofs_global = index_map.size_global

# Print the total number of DOFs
print(f"Total number of DOFs: {num_dofs_global}")

u, p_tr = TrialFunctions(V)
v, p_test = TestFunctions(V)

# Discontinuous function space for material properties
Q_element = functionspace(domain, ("DG", 0))
rho=Function(Q_element)
mu=Function(Q_element)
kappa=Function(Q_element)


# Get indices for each tissue category
wm_index     = cell_tags.find(1)
gm_index     = cell_tags.find(2)
csf_index    = cell_tags.find(3)
bone_index   = cell_tags.find(4)
scalp_index  = cell_tags.find(5)
muscle_index = cell_tags.find(6)

variation = Function(Q_element)
def read_function(filename: Path, timestamp: float, funcname):

    import adios4dolfinx

    in_mesh = adios4dolfinx.read_mesh(filename, MPI.COMM_WORLD)
    W = fem.functionspace(in_mesh, ("DG", 0))
    u_in = fem.Function(W)
    adios4dolfinx.read_function(filename, u_in, time=timestamp, name=funcname)
    return u_in


padding = 1e-14
fine_mesh_cell_map = domain.topology.index_map(domain.topology.dim)
num_cells_on_proc = fine_mesh_cell_map.size_local + fine_mesh_cell_map.num_ghosts
cells = np.arange(num_cells_on_proc, dtype=np.int32)

# heterogeneous stiffness data from MRE of Hiscox's paper
stiffnessh = read_function(str(SCRIPT_DIR / "storage_field_affine_rotated.bp"), 0.0, funcname="stiffness")
VstiffnessR = stiffnessh.function_space

interpolation_data_stiffness = create_interpolation_data(
    Q_element, VstiffnessR, cells, padding=padding
)

stiffness = fem.Function(Q_element)
stiffness.interpolate_nonmatching(
    stiffnessh, cells, interpolation_data=interpolation_data_stiffness
)
# Calculate the mean stiffness
# Replace NaNs/Infs with 0 in-place
np.nan_to_num(stiffness.x.array[:], nan=0.0, posinf=0.0, neginf=0.0, copy=False)
# Gather values
stiffness_wm = domain.comm.gather(stiffness.x.array[wm_index], root=0)
stiffness_gm = domain.comm.gather(stiffness.x.array[gm_index], root=0)

if domain.comm.rank == 0:
    stiffness_wm = np.concatenate(stiffness_wm)
    stiffness_gm = np.concatenate(stiffness_gm)

    # Exclude zeros when computing mean
    combined = np.concatenate((stiffness_wm, stiffness_gm))
    nonzero = combined[combined != 0]

    if nonzero.size > 0:
        mean_value = np.mean(nonzero)
    else:
        mean_value = 0.0 + 0.0j  # fallback if all values were zero

    # Double-check
    wm_nans = np.isnan(stiffness_wm).sum()
    gm_nans = np.isnan(stiffness_gm).sum()
else:
    mean_value = None

mean_value = domain.comm.bcast(mean_value, root=0)

variation.x.array[wm_index] = (stiffness.x.array[wm_index] - mean_value) / mean_value
variation.x.array[gm_index] = (stiffness.x.array[gm_index] - mean_value) / mean_value
variation.x.array[csf_index] = np.full_like(csf_index, 0, dtype=default_scalar_type)
variation.x.array[bone_index] = np.full_like(bone_index, 0, dtype=default_scalar_type)
variation.x.array[muscle_index] = np.full_like(
    muscle_index, 0, dtype=default_scalar_type
)
variation.x.array[scalp_index] = np.full_like(scalp_index, 0, dtype=default_scalar_type)
tol = 1e-12
tol_ = 0.5
mask_ = np.isclose(variation.x.array.real, -1.0, atol=tol_) & np.isclose(
    variation.x.array.imag, 0.0, atol=tol
)
variation.x.array[mask_] = 0 + 0j
variation.x.scatter_forward()

# define dirichlet
fix_facets = locate_entities_boundary(domain, domain.topology.dim - 1, fix_face)
fix_facets.sort()

absorbing_facets = locate_entities_boundary(domain, domain.topology.dim - 1, absorbing_face)
absorbing_facets.sort()

u_bc = fem.Constant(domain, default_scalar_type(0.0))
dofs_bottom_x = locate_dofs_topological(V.sub(0).sub(0), 2, fix_facets)
dofs_bottom_y = locate_dofs_topological(V.sub(0).sub(1), 2, fix_facets)
dofs_bottom_z = locate_dofs_topological(V.sub(0).sub(2), 2, fix_facets)
bcs_1 = dirichletbc(u_bc,dofs_bottom_x,V.sub(0).sub(0))
bcs_2 = dirichletbc(u_bc,dofs_bottom_y,V.sub(0).sub(1))
bcs_3 = dirichletbc(u_bc,dofs_bottom_z,V.sub(0).sub(2))

# define the loading surface
loading_facet = locate_entities_boundary(domain, domain.topology.dim - 1, load_face)
loading_facet.sort() # this is important when applying the neumann boundary!!!!!

dofs_top_x = locate_dofs_topological(V.sub(0).sub(0), 2, loading_facet)
dofs_top_y = locate_dofs_topological(V.sub(0).sub(1), 2, loading_facet)
dofs_top_z = locate_dofs_topological(V.sub(0).sub(2), 2, loading_facet)
bcs_top_x = dirichletbc(u_bc, dofs_top_x, V.sub(0).sub(0))
bcs_top_y = dirichletbc(u_bc, dofs_top_y, V.sub(0).sub(1))
bcs_top_z = dirichletbc(u_bc, dofs_top_z, V.sub(0).sub(2))


# define the stiffness components for each of the index
kappa.x.array[wm_index] = np.full_like(wm_index, kap_wm, dtype=default_scalar_type)* (1+variation.x.array[wm_index])
kappa.x.array[gm_index] = np.full_like(gm_index, kap_gm, dtype=default_scalar_type)* (1+variation.x.array[gm_index])
kappa.x.array[csf_index] = np.full_like(csf_index, kap_csf, dtype=default_scalar_type)
kappa.x.array[bone_index] = np.full_like(bone_index, kap_b, dtype=default_scalar_type)
kappa.x.array[muscle_index] = np.full_like(muscle_index, kap_m, dtype=default_scalar_type)
kappa.x.array[scalp_index] = np.full_like(scalp_index, kap_s, dtype=default_scalar_type)
kappa.x.scatter_forward()

mu.x.array[wm_index] = np.full_like(wm_index, G_wm, dtype=default_scalar_type)
mu.x.array[gm_index] = np.full_like(gm_index, G_gm, dtype=default_scalar_type)
mu.x.array[csf_index] = np.full_like(csf_index, G_csf, dtype=default_scalar_type)
mu.x.array[bone_index] = np.full_like(bone_index, G_b, dtype=default_scalar_type)
mu.x.array[muscle_index] = np.full_like(muscle_index, G_m, dtype=default_scalar_type)
mu.x.array[scalp_index] = np.full_like(scalp_index, G_s, dtype=default_scalar_type)
mu.x.scatter_forward()


rho.x.array[wm_index] = np.full_like(wm_index, rho_wm, dtype=default_scalar_type)
rho.x.array[gm_index] = np.full_like(gm_index, rho_gm, dtype=default_scalar_type)
rho.x.array[csf_index] = np.full_like(csf_index, rho_csf, dtype=default_scalar_type)
rho.x.array[bone_index] = np.full_like(bone_index, rho_b, dtype=default_scalar_type)
rho.x.array[muscle_index] = np.full_like(muscle_index, rho_m, dtype=default_scalar_type)
rho.x.array[scalp_index] = np.full_like(scalp_index, rho_s, dtype=default_scalar_type)
rho.x.scatter_forward()


# spatial coordinate (kept for parity with the forward-source script)
x_ = ufl.SpatialCoordinate(domain)

# absorbing face_mark

facets_indices,facets_markers = [],[]
facets_indices.append(absorbing_facets)
facets_indices.append(loading_facet)
facets_markers.append(np.full_like(absorbing_facets,100000))
facets_markers.append(np.full_like(loading_facet,200000))
facets_indices = np.hstack(facets_indices).astype(np.int32)
facets_markers = np.hstack(facets_markers).astype(np.int32)
sorted_facets = np.argsort(facets_indices)

ft = mesh.meshtags(domain, domain.topology.dim-1, facets_indices[sorted_facets], facets_markers[sorted_facets])
ds = ufl.Measure('ds', domain=domain, subdomain_data=ft)

# Spatial dimension
d = 3
# Identity tensor
I = ufl.variable(ufl.Identity(d))

def epsilon(u_):
    return ufl.sym(ufl.grad(u_))  # Equivalent to 0.5*(ufl.nabla_grad(u) + ufl.nabla_grad(u).T)

def sigma(u_, p_):
    Id = ufl.Identity(domain.geometry.dim)
    return -p_ * Id + 2 * mu * (epsilon(u_) - 1 / 3 * ufl.tr(epsilon(u_)) * Id)


# absorbing boundary condition parameters
# wavespeed, for the absorbing boundary condition
c_p=ufl.sqrt((kappa+4/3*mu)/rho)
c_s=ufl.sqrt(mu/rho)

n = ufl.FacetNormal(domain)
M1 = (
    rho * c_p * ufl.outer(n, n) +
    rho * c_s * (ufl.Identity(domain.geometry.dim) - ufl.outer(n, n))
)

# phase conjugation loaded boundary
V_sub_u = V.sub(0)
V_sub_collapsed_u = V_sub_u.collapse()[0]

V_sub_p = V.sub(1)
V_sub_collapsed_p = V_sub_p.collapse()[0]

results_folder = SCRIPT_DIR / "results_hete_deep_0.96"
results_folder.mkdir(exist_ok=True, parents=True)
uh_array = np.load(
    str(results_folder / "brain_uh_mpi_{}.npy".format(MPI.COMM_WORLD.rank))
)
ph_array = np.load(
    str(results_folder / "brain_ph_mpi_{}.npy".format(MPI.COMM_WORLD.rank))
)

uh_ = Function(V_sub_collapsed_u)
ph_ = Function(V_sub_collapsed_p)

uh_.x.array[:] = uh_array
ph_.x.array[:] = ph_array
reaction_form = ufl.inner(ufl.conj(sigma(uh_, ph_) * n), v) * ds(200000)
bcs_new = [bcs_1, bcs_2, bcs_3]

F_new = (
    omega**2 * rho * ufl.inner(u, v) * ufl.dx
    - (ufl.inner(sigma(u, p_tr), grad(v))) * ufl.dx
    - 1j * omega * ufl.inner(ufl.dot(M1, u), v) * ds(100000)
    + ufl.inner((ufl.div(u) + p_tr / kappa), p_test) * ufl.dx
    + reaction_form
)

a_new = ufl.lhs(F_new)
L_new = ufl.rhs(F_new)

problem_new = LinearProblem(
    a_new,
    L_new,
    bcs=bcs_new,
    petsc_options={
        "ksp_type": "fgmres",
        "pc_type": "lu",
        "pc_precision": "single",
        "pc_factor_mat_solver_type": "mumps",
        "mat_mumps_icntl_35": 1,   # BLR off: CBs stay uncompressed under BLR, which
                                   # blocks most of the OOC memory benefit (see log:
                                   # ICNTL(37) BLR CB compression eff. choice = 0)
        "mat_mumps_icntl_4": 2,
        "mat_mumps_cntl_7":1e-4,
        "mat_mumps_icntl_28": 2,   # parallel analysis (not sequential)
        "mat_mumps_icntl_29": 2,   # ParMETIS
        #"mat_mumps_icntl_22": 1,   # out-of-core factorization
        "ksp_view": None,
        "ksp_view_final_residual": None,
        "ksp_converged_reason": None,
    },
    petsc_options_prefix="linear_",
)
print("start solving")
w_h_new = problem_new.solve()
w_h_new.x.scatter_forward()

uh_new = w_h_new.sub(0).collapse()
uh_new.x.scatter_forward()

ph_new = w_h_new.sub(1).collapse()
ph_new.x.scatter_forward()

problem_new.solver.destroy()
problem_new.A.destroy()
problem_new.b.destroy()
del(problem_new)
gc.collect()

# post-process
def setup_projection(u, V):
    trial = ufl.TrialFunction(V)
    test = ufl.TestFunction(V)

    a = ufl.inner(trial, test) * dx
    L = ufl.inner(u, test) * dx

    projection_problem = LinearProblem(
        a,
        L,
        petsc_options={
            "ksp_type": "cg",
            "ksp_rtol": 1e-16,
            "ksp_atol": 1e-16,
            "ksp_max_it": 1000,
        },
        petsc_options_prefix="project_",
    )

    return projection_problem

Q_P1 = functionspace(domain, ("Lagrange", 1))

s = sigma(uh_new,ph_new) - 1.0 / 3.0 * ufl.tr(sigma(uh_new,ph_new)) * ufl.Identity(len(uh_new))
von_Mises = ufl.sqrt(3.0 / 2.0 * ufl.inner(s, s))


project_mises = setup_projection(von_Mises, Q_P1)

stresses = project_mises.solve()


# write output
uh_new.name = "Displacement_n"
stresses.name = "Shear wave_n"

np.save(
    str(results_folder / "brain_uh_new_mpi_{}.npy".format(MPI.COMM_WORLD.rank)),
    uh_new.x.array,
)
np.save(
    str(results_folder / "brain_ph_new_mpi_{}.npy".format(MPI.COMM_WORLD.rank)),
    ph_new.x.array,
)
with io.VTXWriter(MPI.COMM_WORLD, str(results_folder / "brain_hete_freq{}_disp_inverse.bp".format(f0)), [uh_new], engine="BP4") as vtx:
    vtx.write(0.0)

stress_file = io.XDMFFile(domain.comm, str(results_folder / "brain_hete_freq{}_stress_inverse.xdmf".format(f0)), "w")
stress_file.write_mesh(domain)
stress_file.write_function(stresses, 0)
stress_file.close()


# Create function space for tensor components
V_tensor = fem.functionspace(domain, ("Lagrange", 1, (3, 3)))

# Create expressions for all strain and stress components
strain_ = epsilon(uh_new)
stress_ = sigma(uh_new,ph_new)

project_strain = setup_projection(strain_, V_tensor)
project_stress = setup_projection(stress_, V_tensor)
strain_components = project_strain.solve()
stress_components = project_stress.solve()

# Write the components to files
strain_t_file = io.XDMFFile(domain.comm, str(results_folder / "brain_hete_freq{}_strain_components_inverse.xdmf".format(f0)), "w")
strain_t_file.write_mesh(domain)
strain_t_file.write_function(strain_components, 0)
strain_t_file.close()

stress_t_file = io.XDMFFile(domain.comm, str(results_folder / "brain_hete_freq{}_stress_components_inverse.xdmf".format(f0)), "w")
stress_t_file.write_mesh(domain)
stress_t_file.write_function(stress_components, 0)
stress_t_file.close()


end_time = time.time()
total_runtime = end_time - start_time
print(f"Total runtime: {total_runtime:.2f} seconds")

end_memory = get_memory_usage()
print(f"Memory usage: {end_memory - start_memory:.2f} MB")
