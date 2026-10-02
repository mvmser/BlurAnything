# Models

| File | Source | Size | Role |
| --- | --- | --- | --- |
| `yolov8n-seg.onnx` | Ultralytics YOLOv8n-seg | 7.0 MB | "Fast" model |
| `yolov8s-seg.onnx` | Ultralytics YOLOv8s-seg | 23.8 MB | "Accurate" model |

Both are produced by [`scripts/export_models.py`](../../scripts/export_models.py) from the official
Ultralytics weights (80 COCO classes, instance segmentation, 640x640 input).
Weights are stored as float16 and cast back to float32 when the session starts, which halves the download
with no measurable accuracy change (see the script for details).

The YOLOv8 weights are © Ultralytics and licensed under the
[AGPL-3.0](https://www.gnu.org/licenses/agpl-3.0.html). See `THIRD_PARTY_NOTICES.md` at the repository root.

After re-exporting, update `bytes` and `sha256` in `web/js/config.js` (a unit test checks them) and bump
`MODEL_CACHE` there.
