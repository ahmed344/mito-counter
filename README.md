# mito-counter
Count the mitocondria in electron microscopic images

## Recut training tiles

If you need to recut image tiles based on the `-LOC-` coordinates embedded in the
filenames while leaving masks untouched, run:

```
python /workspaces/mito-counter/recut_tiles.py \
  --training-root /workspaces/mito-counter/training_data \
  --source-root /workspaces/mito-counter/data/Calpaine_3/Processed
```

## Empanada napari troubleshooting

If the napari plugin is slow or hits CUDA out-of-memory (OOM), it usually means the
plugin is trying to run the full image or very large tiles on the GPU. The same
model can appear fast in a script if it uses smaller inputs, fewer tiles, or
different defaults.

Practical fixes in the napari Empanada panel:
- Set **Tile size** to a non-zero value so the image is processed in patches.
  Start with `512` or `1024` and increase only if GPU memory allows.
- Disable **Batch mode** unless you are intentionally processing many slices.
- Downsample or crop large images before inference (smaller inputs reduce VRAM).
- If GPU memory is limited, switch to CPU or the **quantized model** (CPU-only).

Optional environment help:
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` can reduce fragmentation and
  make large allocations more reliable.
