# Third-party notices

BlurAnything bundles or derives from the following components. Each keeps its own license.

## Machine-learning models

| Component                                                 | Files                                           | License                                                          |
| --------------------------------------------------------- | ----------------------------------------------- | ---------------------------------------------------------------- |
| **YOLOv8-seg** weights by [Ultralytics](https://ultralytics.com) | `web/models/yolov8n-seg.onnx`, `yolov8s-seg.onnx` | [AGPL-3.0](https://www.gnu.org/licenses/agpl-3.0.html) |

The ONNX files are conversions of the official `yolov8n-seg.pt` / `yolov8s-seg.pt` weights, produced by
[`scripts/export_models.py`](scripts/export_models.py) (weights additionally stored as float16). Source code of this
application, which uses them, is available at <https://github.com/mvmser/BlurAnything>.

## Runtime

| Component                                                 | Files                | License                                   |
| --------------------------------------------------------- | -------------------- | ----------------------------------------- |
| **ONNX Runtime Web** 1.30.0, © Microsoft Corporation      | `web/vendor/ort/*`   | MIT (see `web/vendor/ort/LICENSE`)        |

## Images

| File                       | Origin                                                                       | License                                  |
| -------------------------- | ---------------------------------------------------------------------------- | ---------------------------------------- |
| `web/samples/bus.jpg`      | Ultralytics demo image ([assets](https://github.com/ultralytics/assets))     | AGPL-3.0                                 |
| `web/samples/astronaut.jpg`| Eileen Collins, [NASA Great Images](https://flic.kr/p/r9qvLn) (via scikit-image) | Public domain                        |
| `web/samples/cat.jpg`      | "Chelsea the cat", Stefan van der Walt (via scikit-image)                    | CC0                                      |
| `web/samples/coffee.jpg`   | Rachel Michetti, Pikolo Espresso Bar (via scikit-image)                      | CC0                                      |

## Icons

Interface icons are drawn in the style of [Lucide](https://lucide.dev) (ISC license); the GitHub mark comes from
[Simple Icons](https://simpleicons.org) (CC0). The BlurAnything logo is original.

## Development tooling (not shipped)

Playwright (Apache-2.0), axe-core (MPL-2.0), Prettier (MIT).
