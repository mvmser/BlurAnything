# Third-party notices

| What | Where | License |
| --- | --- | --- |
| YOLOv8-seg weights, converted to ONNX (Ultralytics) | `web/models/` | [AGPL-3.0](https://www.gnu.org/licenses/agpl-3.0.html) |
| ONNX Runtime Web 1.30.0 (Microsoft) | `web/vendor/ort/` | MIT, see `web/vendor/ort/LICENSE` |
| `bus.jpg`, the Ultralytics demo image | `web/samples/` | AGPL-3.0 |
| `astronaut.jpg`, Eileen Collins ([NASA](https://flic.kr/p/r9qvLn)) | `web/samples/` | public domain |
| `cat.jpg`, by Stefan van der Walt | `web/samples/` | CC0 |
| `coffee.jpg`, by Rachel Michetti | `web/samples/` | CC0 |

The ONNX files are converted from the official `yolov8n-seg.pt` and `yolov8s-seg.pt` weights with
[`scripts/export_models.py`](scripts/export_models.py). The last three photos come from scikit-image's sample data.

The interface icons are drawn in the style of [Lucide](https://lucide.dev) (ISC), the GitHub mark comes from
[Simple Icons](https://simpleicons.org) (CC0).
