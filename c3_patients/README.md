# Calpain-3 patient image pipeline

This pipeline processes the 4800X electron-microscopy images in:

- `data/Calpaine_3_patients/RAW/CAPN3/*/DATA QUANTIFICATION - IMF`
- `data/Calpaine_3_patients/RAW/CAPN3/*/DATA QUANTIFICATION - SS`
- `data/Calpaine_3_patients/RAW/CTRL/*/DATA QUANTIFICATION - IMF`
- `data/Calpaine_3_patients/RAW/CTRL/*/DATA QUANTIFICATION - SS`

It deliberately excludes `ME DATA` and image-preview directories. Run commands
from the repository root (`/workspaces/mito-counter`) in the devcontainer base
Python environment.

## 1. Convert DM3 images

Preview the selected inputs and output paths:

```bash
python c3_patients/dm3_to_tiff.py --dry-run
```

Convert the DM3 files to 8-bit TIFF files and JSON metadata sidecars:

```bash
python c3_patients/dm3_to_tiff.py
```

The default `4800X` filter can be overridden with
`--magnification-token TOKEN`. Outputs are written below
`data/Calpaine_3_patients/Processed/<condition>/<identifier>/<compartment>/`.
Review the final metadata-alignment report. Patients listed in the clinical
CSVs but lacking image folders are reported and do not create output rows.

## 2. Correct image background

First inspect the planned work:

```bash
python tiff_background_correct.py \
  --input-root /workspaces/mito-counter/data/Calpaine_3_patients/Processed \
  --dry-run
```

Then run correction:

```bash
python tiff_background_correct.py \
  --input-root /workspaces/mito-counter/data/Calpaine_3_patients/Processed
```

Each source TIFF receives a sibling file named `*_corrected.tif`.

## 3. Segment mitochondria

MitoNet requires a CUDA-capable environment. The patient configuration is
separate from the default configuration used by other datasets:

```bash
python mitonet_infenence.py \
  --config c3_patients/mitonet_infenence.yaml
```

For a one-image smoke test, temporarily set `paths.input_file` in
`c3_patients/mitonet_infenence.yaml` to one corrected TIFF path. Set it back to
`null` before the full run. Each corrected image should receive
`*_segmented.tif` and `*_segmented_metrics.csv`.

## 4. Generate segmentation overlays

Generate a small QC sample first:

```bash
python c3_patients/build_segmentation_overlay_qc.py --limit 20
```

Generate overlays for the complete dataset:

```bash
python c3_patients/build_segmentation_overlay_qc.py
```

Existing overlays are skipped. Use `--overwrite` to regenerate them. Outputs
mirror the processed directory tree below
`data/Calpaine_3_patients/results/overlay_qc/`. Inspect examples from CAPN3 and
CTRL, IMF and SS, and multiple patients before accepting the segmentation.

## 5. Build measurements

After all segmentations have been reviewed, run:

```bash
python c3_patients/build_measurements_csv.py
```

The measurement stage stops if any converted image lacks its source TIFF or
segmentation metrics CSV, preventing unprocessed images from being counted as
zero-mitochondria fields.

The script writes:

- `measurements.csv`
- `measurements_cleaned.csv`
- `measurements_cleaned_ss_summary.csv`
- `measurements_cleaned_imf_summary.csv`
- `measurements_cleaned_no_compartment.csv`
- `measurements_cleaned_no_compartment_summary.csv`

under `data/Calpaine_3_patients/results/`.

The cleaned file removes very small objects, objects touching an image edge,
and disconnected segmentations. All clinical CSV fields are retained.
Compartment-specific files use `Intermyofibrillar (IMF)` or
`Sub-sarcolemmal (SS)`. No-compartment files retain the `Compartment` column
with the value `All compartments`. Eccentricity is bounded from 0 to 1 and
replaces elongation in final measurement outputs.

## Completion checks

Before later statistical analysis, confirm:

1. The converter selected only `DATA QUANTIFICATION - IMF` and
   `DATA QUANTIFICATION - SS` images at 4800X.
2. Every TIFF has a JSON sidecar and every corrected TIFF has both segmentation
   outputs.
3. The measurement script reports pixel-size sources and no unexpected
   metadata mismatches.
4. `IDENTIFIER`, condition, biopsy site, compartment, age, gender, genotype,
   and notes match the clinical source CSVs.
5. Overlay QC is acceptable across both conditions and compartments.
6. `Eccentricity` values are between 0 and 1.

Statistical tests and hierarchical Bayesian models are intentionally outside
this pipeline stage.
