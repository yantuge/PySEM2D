# PySEM2D

PySEM2D is a spectral element solver for 2D elastic wave propagation in isotropic and anisotropic media. It supports PML absorbing boundaries, Taichi CPU/GPU acceleration, wavefield snapshots, and receiver seismograms.

## Installation

Use a Python environment supported by Taichi. On **Windows (PowerShell)**, install the dependencies:

```powershell
python -m pip install numpy scipy matplotlib PyYAML taichi gmsh meshio
```

On **Linux/macOS (Bash or Zsh)**, create and activate a virtual environment in the project root, then install the dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install numpy scipy matplotlib PyYAML taichi gmsh meshio
```

Run `source .venv/bin/activate` from the project root whenever you open a new terminal.

The supplied configurations use `solver.taichi.arch: 'cuda'`. For CPU execution, change this value to `'cpu'` in the YAML file you plan to run. Set `visualization.enabled: false` to run without the live visualization window.

Before running on **Linux/macOS**, edit each YAML configuration you will use:

- Use forward slashes in paths; for example, set `output.directory` to `'./results'` instead of `'.\results'` or `'.\results\'`.
- On macOS, set `solver.taichi.arch` to `'cpu'`. On Linux, use `'cpu'` for CPU execution or retain `'cuda'` for an NVIDIA GPU with CUDA support.

## Usage

Open a terminal in the project root (the folder containing `main.py`). On **Windows (PowerShell)**, run the default configuration:

```powershell
python main.py
```

This loads `config/config.yaml`. To select another configuration or view command-line help:

```powershell
python main.py -c config/my_config.yaml
python main.py --help
```

On **Linux/macOS (Bash or Zsh)**, use `python3`:

```bash
python3 main.py                           # Default: config/config.yaml
python3 main.py -c config/my_config.yaml  # Custom configuration
python3 main.py --help                   # Command-line help
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

On **Linux/macOS (Bash or Zsh)**, start from the project root with the virtual environment activated and the YAML settings adjusted as described above:

```bash
# Case 1: Isotropic rectangular model
(cd verification/case1/PySEM2D && python3 ../../../main.py -c Case1_verification.yaml)

# Case 2: Anisotropic (VTI) rectangular model
(cd verification/case2/PySEM2D && python3 ../../../main.py -c Case2_verification.yaml)

# Case 3: Isotropic mountain model
(cd verification/case3/PySEM2D && python3 ../../../main.py -c Case3_verification.yaml)

# Case 4: Isotropic model with multiple material regions
(cd verification/case4/PySEM2D && python3 ../../../main.py -c Case4_verification.yaml)

# Case 5: Anisotropic model with multiple material regions
(cd verification/case5/PySEM2D && python3 ../../../main.py -c Case5_verification.yaml)
```

The parentheses run each case in a subshell, so your terminal remains in the project root after each command. The `../../../main.py` path points to the main entry file from the case's `PySEM2D` directory.

These commands run the PySEM2D simulations. The `SPECFEM2D` folders in cases 1, 2, 3, and 5 contain comparison inputs; case 1 also includes analytical reference data in `REF_ANALYTICAL`.

## Results

Results are saved to `output.directory` in the selected YAML file. Outputs include `model_setup.png`, receiver station files, seismogram data and plots, and VTU wavefield snapshots when enabled. Open the snapshots in ParaView to inspect wave propagation.
