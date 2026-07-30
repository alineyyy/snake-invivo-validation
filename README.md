# SNAKE In-Vivo Validation

This directory collects the scripts used to validate SNAKE reconstruction on
simulated and in-vivo 3 T fMRI data. The workflow has three stages:

1. generate simulated R1/R2/R4/R8 datasets;
2. reconstruct Siemens in-vivo acquisitions on the TGCC SLURM GPU cluster;
3. run first-level GLM analysis, activation-map QC, and F2 scoring.

The Bash files in `simulation/` and `invivo_reconstruction/` are designed to
run on the TGCC supercomputer through SLURM. They use TGCC-specific modules,
accounts, partitions, `$WORK`, and `/ccc/scratch` paths. The Python and notebook
components are also available in the local `/volatile/Caini` workspace.

## Directory structure

```text
snake_invivo_validatation/
├── README.md
├── simulation/
│   ├── params_r1.txt
│   ├── params.txt
│   ├── simulate_r1.sh
│   └── simulate_acceleration.sh
├── invivo_reconstruction/
│   ├── reconstruct_batch.sh
│   ├── recon_script_3T.py
│   └── siemens_loader.py
└── glm_analysis/
    ├── GLM_analysis.ipynb
    ├── glm_activation_workflow.py
    ├── fmri_glm_toolkit.py
    └── fmri_metric_toolkit.py
```

Generated `__pycache__` directories are not part of the workflow.

## Requirements

The complete pipeline expects:

- Python 3.10;
- the `snake` and `snake-fmri` source trees;
- NumPy, SciPy, pandas, matplotlib, nibabel, nilearn, scikit-learn,
  statsmodels, Nipype, and ipywidgets;
- PySAP/MRI reconstruction dependencies required by SNAKE;
- CUDA 12.2 and cuDNN 8.9.7 for the cluster jobs;
- SPM standalone with MATLAB Runtime R2024b for realignment;
- access to the configured TGCC `$WORK` and `/ccc/scratch` paths.

The GLM notebook currently uses modules stored beside the notebook, so it
should be opened from `glm_analysis/` or with that directory on `PYTHONPATH`.

## 1. Simulation

The simulation stage uses the SNAKE Hydra CLI. Two parameter tables define the
SLURM array jobs. Both simulation Bash scripts are submitted and executed on
TGCC compute nodes.

### R1 reference simulation

`simulation/params_r1.txt` has one row per reference subject:

```text
job_name  acsz  accelz  init_strategy  id  SNR
```

The current table contains four R1 cases. Submit it with:

```bash
cd /volatile/Caini/snake_invivo_validatation/simulation
sbatch simulate_r1.sh
```

`simulate_r1.sh` uses the `scenario2-3T_r1` configuration, one coil, a constant
stack-of-spiral trajectory, and the SNR value from each table row.

### Accelerated simulation

`simulation/params.txt` defines cold- and global-initialized R2/R4/R8 cases:

```bash
cd /volatile/Caini/snake_invivo_validatation/simulation
sbatch simulate_acceleration.sh
```

`simulate_acceleration.sh` uses `scenario2-3T` and a time-varying
stack-of-spiral trajectory. Its SLURM array range must match the number of data
rows in `params.txt` (currently 24 rows, therefore `--array=0-23`).

The accelerated simulation uses the SNR configuration defined by the submitted
script and parameter table.

## 2. In-vivo reconstruction

`invivo_reconstruction/reconstruct_batch.sh` is the TGCC batch entry point for
in-vivo reconstruction. It is submitted with `sbatch` and runs on the TGCC A100
partition using the SLURM directives declared at the beginning of the file.

`invivo_reconstruction/reconstruct_batch.sh` splits every reconstruction
setting into multiple frame chunks and runs them as a SLURM array. The run-list
uses the following whitespace-separated columns:

```text
job_name trajectory_file data_file shots_per_frame total_frames reconstructor init_strategy restart_strategy is_time_varying
```

For each run-list row, the script:

1. resolves the raw Siemens `.dat`, trajectory, and sensitivity-map paths;
2. computes the frame range assigned to the current array task;
3. calls `recon_script_3T.py` for that frame range;
4. writes reconstruction chunks and logs below the configured output root.

The cluster paths used by the job are configured near the top of
`reconstruct_batch.sh`:

- `DATA_ROOT`
- `TRAJ_ROOT`
- `SMAPS_FILE`
- `OUT_ROOT`
- `RUNLIST`
- `$WORK/codes_3T`
- `$WORK/recon_pipeline`

Submit the reconstruction array with:

```bash
cd /volatile/Caini/snake_invivo_validatation/invivo_reconstruction
sbatch reconstruct_batch.sh
```

The script automatically maps each SLURM array index to a reconstruction
parameter set and frame chunk. Empty chunks exit cleanly, while valid chunks
write their reconstruction arrays and logs under `OUT_ROOT`.

## 3. GLM and activation-map analysis

Open the notebook from the analysis directory:

```bash
cd /volatile/Caini/snake_invivo_validatation/glm_analysis
jupyter lab GLM_analysis.ipynb
```

The notebook is organized into the following sections:

1. imports;
2. global output, T1, SPM, and affine configuration;
3. reconstruction input specifications;
4. first-level GLM execution;
5. activation-map QC and the optional interactive QC viewer;
6. F2 reference visits and anchor masks;
7. F2 computation;
8. CSV/figure export;
9. optional comparison with simulation results.

Before running the notebook, verify:

- `OUTPUT_ROOT` and `DEFAULT_T1`;
- the visit-specific affine;
- every `ReconSpec` path, date glob, scan count, TR, and scale;
- the SPM standalone command;
- `F2_REFERENCE_T1`, `F2_BASES`, and `ANCHORS_BY_DATE`;
- the optional simulation F2 CSV path.

### GLM helper modules

- `glm_activation_workflow.py` stores visit-specific affines and provides
  `make_config`, `make_specs`, and `preview_specs` helpers.
- `fmri_glm_toolkit.py` loads reconstruction chunks, creates NIfTI files, runs
  SPM realignment and first-level GLM analysis, and generates QC figures.
- `fmri_metric_toolkit.py` creates activation masks, computes Dice/F2 metrics,
  summarizes visits and reconstruction methods, and provides diagnostic plots.

## F2 score definition

For a predicted activation mask and a reference union mask, the F-beta score is
computed with `beta=2`. This weights recall more strongly than precision. The
reference mask for each visit is the union of the two configured R1 anchor
masks.

The notebook exports:

```text
<OUTPUT_ROOT>/f2_analysis/f2_scores.csv
<OUTPUT_ROOT>/f2_analysis/f2_scores.png
```

The CSV contains one row per reconstruction case. The figure summarizes F2 by
acceleration factor and reconstruction method using the standard error of the
mean.

## Expected data flow

```text
Simulation parameter tables ──> SNAKE simulation outputs

Siemens .dat + trajectory + sensitivity maps
                    │
                    v
           reconstruction chunks (.npy)
                    │
                    v
       NIfTI conversion + SPM realignment
                    │
                    v
          first-level GLM + z-maps
                    │
                    v
       activation QC + Dice/F2 analysis
```

## Reproducibility checklist

Before reporting validation results, record:

- the Git revisions of `snake`, `snake-fmri`, and reconstruction dependencies;
- the exact simulation parameter tables;
- raw-data, trajectory, and sensitivity-map identifiers;
- reconstruction method, initialization strategy, number of iterations, and
  density-compensation/ORC settings;
- visit affine, T1 reference, TR, scan count, and GLM threshold settings;
- the two R1 anchors used to create each F2 reference mask;
- SLURM job IDs and output directories.

## Notes

- The directory name uses the spelling `validatation`; scripts and documentation
  retain that path for compatibility.
- Most configured paths are absolute. Copying this directory alone is not
  sufficient to reproduce the workflow on a different machine.
- Run the expensive SLURM, SPM, and F2 stages only after validating all input
  paths with lightweight checks.
