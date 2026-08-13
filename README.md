# Modeling transcranial ultrasound waves in heterogeneous brain

FEniCSx/dolfinx pipeline for simulating elastic wave focusing in a
heterogeneous head model built from the
[MIDA](https://itis.swiss/virtual-population/regional-human-models/mida-model/)
atlas, with subject-specific MRE storage modulus maps mapped onto white and
gray matter to create a heterogeneous bulk modulus field.

## Pipeline

1. **`niiread_group.py`** — collapses the MIDA atlas's ~100 fine-grained
   tissue labels into 6 coarse groups (white matter, gray matter, CSF, bone,
   scalp, muscle) and crops to the region of interest. Requires `MIDA_v1.nii`
   (see [Sources](#sources)). Produces `MIDA_roi_uint8_group_full.nii`.

   *(Meshing that ROI volume into `input/MIDA_processed_half_group_0.96.msh`,
   used by the scripts below, is done externally with
   [CGAL](https://github.com/cgal/cgal) and [Gmsh](https://gmsh.info/) — not
   included in this repo.)*

2. **`cube_hete_mre.py`** — reads subject MRE stiffness/damping NIfTI volumes
   (`MRE134_Stiffness3D.nii`, `MRE134_Damping3D.nii`), re-orients them into
   the head mesh's world coordinates, and interpolates a storage-modulus
   field onto a regular box mesh. Writes `storage_field_affine_rotated.bp`
   using [adios4dolfinx](https://github.com/jorgensd/adios4dolfinx).

3. **`mida_hete_deep_first.py`** — forward MRE simulation: solves the
   mixed displacement/pressure harmonic wave equation on the full head mesh
   with a Gaussian point source, using per-tissue
   material properties plus the MRE-derived heterogeneity in WM/GM from step
   2. Saves displacement/pressure solution vectors and writes result fields
   (XDMF/VTX) to `results_hete_deep_0.96/`. Built on
   [FEniCSx/dolfinx](https://github.com/FEniCS/dolfinx) for the finite-element
   assembly and solve.

4. **`mida_hete_deep_sec.py`** — phase-conjugation (time-reversal)
   step: rebuilds the same material fields, then re-solves with a
   phase-conjugated reaction load built from step 3's saved solution on the
   loading boundary. Must run after step 3.

## Requirements

Developed and run on an HPC cluster, with `fenics-dolfinx`
built in a [Spack](https://spack.io/) environment (pinned at **0.10.0**):

- `fenics-dolfinx` 0.10.0 (FEniCSx, built against a **complex-mode PETSc**
  — required for the complex-valued harmonic wave equation solved here —
  with MUMPS support, mpi4py, basix, ufl)
- `adios4dolfinx` 0.10.0
- numpy, scipy, nibabel, psutil

## Data

Subject MRI/MRE NIfTI files (`MIDA_v1.nii`, `MRE134_Stiffness3D.nii`,
`MRE134_Damping3D.nii`) and the meshed `input/*.msh` are not included in this
repository — see [Sources](#sources) below for where to get them.

## Sources

- MIDA head model: https://itis.swiss/virtual-population/regional-human-models/mida-model
- CGAL: https://github.com/cgal/cgal
- MRE data: https://github.com/mechneurolab/mre134
- Gmsh: https://gmsh.info/
- FEniCSx/dolfinx: https://github.com/FEniCS/dolfinx
- adios4dolfinx: https://github.com/jorgensd/adios4dolfinx