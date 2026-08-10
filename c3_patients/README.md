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

Convert the DM3 files to 8-bit TIFF files and JSON metadata sidecars. Existing
8-bit pixels remain unchanged. Higher-depth images are linearly mapped from
their individual minimum and maximum to `0–255`, preserving each histogram's
overall shape without percentile clipping:

```bash
python c3_patients/dm3_to_tiff.py
```

The default `4800X` filter can be overridden with
`--magnification-token TOKEN`. Outputs are written below
`data/Calpaine_3_patients/Processed/<condition>/<identifier>/<compartment>/`.
Review the final metadata-alignment report. Patients listed in the clinical
CSVs but lacking image folders are reported and do not create output rows.

## 2. Correct image background

Background correction is configured in `tiff_background_correct.yaml`. Its
default input root points to this patient dataset. Gaussian flat-field
correction and contrast enhancement can be enabled independently in the YAML,
and every setting can be overridden from the command line.

First inspect the planned work:

```bash
python tiff_background_correct.py --dry-run
```

Then run correction:

```bash
python tiff_background_correct.py
```

Each source TIFF receives a sibling file named `*_corrected.tif`.
Corrected TIFFs preserve the source image dtype and bit depth. After optional
Gaussian correction, select at most one contrast method: CLAHE, exact min-max
stretching, or percentile stretching. Percentile bounds and the stretching
output range (default `0–255`) are configurable in the YAML. Values outside
the percentile bounds are clipped to the output endpoints.

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

### Multi-scale segmentation

`paths.downsample_factor` accepts either one number or a list. A list runs one
inference pass per factor and fuses the passes into a single instance
segmentation, because no single factor segments every mitochondrion size well:
factor 1 resolves the smallest mitochondria, factor 4 keeps the largest ones
whole, and factor 2 is the best single-pass compromise. The patient config uses
`[1.0, 2.0, 4.0]`. Runtime and GPU memory scale with the number of factors.

Instances from different passes are treated as the same object when their IoU
reaches `fusion.iou_threshold`, or when `fusion.containment_threshold` of the
smaller instance lies inside the larger one. The containment rule absorbs the
fragments a fine scale produces for one large mitochondrion. Each fused object
is then represented by the finest scale that segmented it as a single instance
covering at least `fusion.min_coverage_ratio` of its largest extent across
scales, so small mitochondria keep the sharp factor-1 boundary while large ones
fall back to factor 2 or 4 instead of staying fragmented.

`fusion.min_votes` controls how many scales must detect an object for it to be
kept. `1` keeps the full union of all passes. Prefer `2` unless the union has
been checked in overlay QC: on low-contrast fields where the finer scales
correctly find nothing, factor 4 can outline whole myofibril blocks as
mitochondria, and a union imports every one of those false positives. Because
counts, nearest-neighbor distances, and Voronoi areas all depend on this
setting, use one value for the entire dataset and record which one.

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
