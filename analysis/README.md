# Paper analysis

This folder produces the figures, tables and numbers of the paper from the
TRENDY model output and the emulator output of `scripts/reproduce_paper.sh`.
The analysis was developed in a notebook: `cells/` holds its cells (file names
keep the cell numbers) and `steps/` the analysis steps run after them.
`run_analysis.py` executes them job by job in one shared namespace, as the
notebook did.

## Inputs

Paths are set in `config.py`; each can be overridden with an environment
variable of the same name.

| Name | Content |
| --- | --- |
| `TRENDY_V13_NBP` | TRENDY v13 S2 monthly NBP of the 19 models, 0.5°, one file with a `DGVM` dimension (the targets of the emulators) |
| `TRENDY_V14_NBP` | TRENDY v14 S2 monthly NBP, same layout (truth of the 2024 test) |
| `EMULATOR_EXP1_DIR` | `result/exp1_recursive/` (experiment 1: one-step rows 2010-2021, recursive 2022-2023) |
| `EMULATOR_EXP1_ONESTEP_DIR` | `result/exp1_onestep/` |
| `EMULATOR_EXP2_DIR` | `result/exp2_recursive/` (experiment 2: one-step rows 2010-2023, recursive 2024) |
| `RECCAP_MASK` | RECCAP2 land regions on the 0.5° grid (`Land_reccap_regions_mask_05x05.nc`, variable `mask`, ids 1-25) |
| `EVENT_MASK` | the 36 excluded cells of western Venezuela (made by `event_mask/make_event_mask.py`) |
| `ERA5_FILE` | ERA5 monthly `t2m` and `tp` on the 0.5° grid (ERA5 regression baseline) |
| `DATA_FEATHER` | the emulator input table (ERA5 columns, model-difference analysis) |
| `OUT_DIR` | output folder (default `output/`) |

Make the event mask first:

```bash
python analysis/event_mask/make_event_mask.py data/analysis/crujra_v2.4_2021_precip_event_mask.nc
```

It writes the cells listed in `event_mask/event_mask_cells.txt`, where the
CRU-based forcing shows a decrease in precipitation in 2021 that is absent from
ERA5. With `--crujra DIR` it recomputes them from the CRU-JRA v2.4 half-degree
forcing files (2021 precipitation below half of the 2010-2020 mean, 8-connected
region); this gives the same 36 cells.

## Running

```bash
python analysis/run_analysis.py all        # or one or more jobs, e.g. figures tables
```

| Job | Outputs (in `OUT_DIR`) | Paper |
| --- | --- | --- |
| `figures` | `Fig2`-`Fig6`, `SuppFig1`, `SuppFig3`-`SuppFig9` (PNG at 600 dpi and PDF); `Fig3_R2_two_scales.txt`, `Fig5_spatial_R2.txt`, `Fig6_tail_underestimation.txt`; `valid_mask_final.nc` | Figs 2-6, Supp Figs 1, 3-9 |
| `tables` | `SuppTable4.csv`, `SuppTable5.csv` | Supp Tables 4, 5 |
| `table6` | `SuppTable6.csv`; `checks/C2_*` (subset-threshold sensitivity) | Supp Table 6 |
| `regions` | `fit_for_purpose/P1_*`, `P2_*`, `P3_*`; `SuppFig2_values.csv` | regional results |
| `climatology` | `fit_for_purpose/P4_*` | climatology baseline |
| `model_differences` | `model_differences/` (needs `figures` and `tables`) | differences between models |
| `onestep` | `checks/C4_onestep_vs_recursive_by_model.csv`, `checks/grid_seasonal_cycle_climatology_R2.csv`, `checks/identity_recursive_vs_onestep_2010_2021.csv` | one-step vs recursive |
| `key_numbers` | `key_numbers.txt` | numbers in the text |
| `supp_table3` | `SuppTable3.csv` (needs the jobs above) | Supp Table 3 |
| `supp_fig2` | `SuppFig2` (needs `regions`) | Supp Fig 2 |
| `source_data` | `source_data/`: CSV source data of the figures and tables (needs the jobs above) | data underlying the figures |

Each job loads the monthly model and emulator fields of all 19 models, so it
needs a machine with a large memory. The log of every step is written to
`OUT_DIR/logs/`.

Conventions used throughout (`cells/c03_definitions.py`): months have their
actual number of days; R² is the squared Pearson correlation; anomalies are
taken against each dataset's own 2010-2021 monthly climatology; every model and
its emulator are evaluated on the same cells (`valid_mask_final.nc`: complete
over 2010-2023, temporal standard deviation above 1e-10 kgC m⁻² s⁻¹, and
outside the event mask).

## Supp Fig 10

`supp_fig10/` draws precipitation, fire emissions and vegetation carbon in the
36 excluded cells: annual precipitation from ERA5, CRU TS 4.08 and CRU-JRA v2.3
and v2.4, and fire emissions and vegetation carbon of LPX-Bern and LPJwsl /
LPJ-EOSIM in TRENDY v11-v14.

```bash
# where the CRU-JRA half-degree forcing files are (v2.3/halfdeg, v2.4/halfdeg)
python analysis/supp_fig10/crujra_precip.py CRUJRA_DIR crujra_precip.json
# ERA5, CRU TS 4.08 and the TRENDY files; paths in a JSON file (see supp_fig10_inputs.example.json)
python analysis/supp_fig10/extract_inputs.py supp_fig10_inputs.json extract_inputs.json
python analysis/supp_fig10/plot_supp_fig10.py crujra_precip.json extract_inputs.json output/SuppFig10
```

## Emulated NBP files

`export_emulated_nbp.py` writes the emulated monthly NBP of the paper as NetCDF
(experiment 1, 2022-2023; experiment 2, 2024; 19 models, 0.5°) with the
evaluation mask:

```bash
python analysis/export_emulated_nbp.py output/emulated_nbp
```

## Software

The paper was produced with Python 3.9 and the package versions in
`requirements.txt` (`pip install -r analysis/requirements.txt`). The figures
use the Arial font, which must be installed for matplotlib.
`cells/c03_definitions.py` corrects how matplotlib 3.8 measures Arial's
x-height in mathtext; other matplotlib versions may place superscripts slightly
differently. Fig 1 is a schematic and is not produced by code.
