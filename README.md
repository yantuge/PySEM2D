# PySEM2D

PySEM2D is a spectral element solver for 2D elastic wave propagation in isotropic and anisotropic media. It supports PML absorbing boundaries, Taichi CPU/GPU acceleration, wavefield snapshots, and receiver seismograms.

## Installation

Use a Python environment supported by Taichi and install the dependencies:

```powershell
python -m pip install numpy scipy matplotlib PyYAML taichi gmsh meshio
```

The supplied configurations use `solver.taichi.arch: 'cuda'`. For CPU execution, change this value to `'cpu'` in the YAML file you plan to run. Set `visualization.enabled: false` to run without the live visualization window.

## Usage

Open a terminal in the project root (the folder containing `main.py`). Run the default configuration:

```powershell
python main.py
```

This loads `config/config.yaml`. To select another configuration or view command-line help:

```powershell
python main.py -c config/my_config.yaml
python main.py --help
```

Create `config/my_config.yaml` by copying and editing `config/config.yaml`. The `--config` option is equivalent to `-c`.

The YAML file defines the time step and number of steps, polynomial order, mesh, material properties, PML boundaries, sources, receivers, output, and solver settings. Mesh input can be Gmsh (`mesh.format: 'gmsh'`, with `mesh.msh_file`) or Abaqus (`mesh.format: 'abaqus'`, with `mesh.inp_file`); Abaqus meshes are converted automatically.

**Relative mesh and output paths are resolved from the terminal's working directory.** Use the project root for the default configuration and each case's `PySEM2D` folder for the verification configurations below.

## Verification examples

Run these commands in **Windows PowerShell**, starting from the project root. Each block returns to the root after running its case.

```powershell
# Case 1: Isotropic rectangular model
Push-Location .\verification\case1\PySEM2D
python ..\..\..\main.py -c Case1_verification.yaml
Pop-Location

# Case 2: Anisotropic (VTI) rectangular model
Push-Location .\verification\case2\PySEM2D
python ..\..\..\main.py -c Case2_verification.yaml
Pop-Location

# Case 3: Isotropic mountain model
Push-Location .\verification\case3\PySEM2D
python ..\..\..\main.py -c Case3_verification.yaml
Pop-Location

# Case 4: Isotropic model with multiple material regions
Push-Location .\verification\case4\PySEM2D
python ..\..\..\main.py -c Case4_verification.yaml
Pop-Location

# Case 5: Anisotropic model with multiple material regions
Push-Location .\verification\case5\PySEM2D
python ..\..\..\main.py -c Case5_verification.yaml
Pop-Location
```

These commands run the PySEM2D simulations. The `SPECFEM2D` folders in cases 1, 2, 3, and 5 contain comparison inputs; case 1 also includes analytical reference data in `REF_ANALYTICAL`.

## Results

Results are saved to `output.directory` in the selected YAML file. Outputs include `model_setup.png`, receiver station files, seismogram data and plots, and VTU wavefield snapshots when enabled. Open the snapshots in ParaView to inspect wave propagation.
