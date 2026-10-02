"""Export the YOLOv8 segmentation models used by BlurAnything to ONNX.

The web app runs the models in the browser with ONNX Runtime Web, so the
PyTorch weights from Ultralytics have to be converted once:

    python -m venv .venv && source .venv/bin/activate
    pip install ultralytics onnx onnxslim onnxruntime
    python scripts/export_models.py            # -> web/models/*.onnx

Options:
    --models yolov8n-seg yolov8s-seg   which weights to export
    --fp16-weights                     store weights as float16 (half the download,
                                       compute stays float32 - see below)

Why --fp16-weights: the .onnx file is downloaded by every visitor. Storing the
constant weights as float16 and casting them back to float32 right after
loading halves the file size while the graph computes exactly like the float32
export (the cast is constant-folded by ONNX Runtime when the session starts).
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "web" / "models"
IMGSZ = 640
OPSET = 12  # widely supported by ONNX Runtime Web's WASM backend
MIN_FP16_ELEMENTS = 1024  # leave tiny tensors (biases, shapes) alone


def export_onnx(name: str, out_dir: Path) -> Path:
    from ultralytics import YOLO

    model = YOLO(f"{name}.pt")  # weights are fetched from the Ultralytics assets release
    exported = Path(
        model.export(format="onnx", imgsz=IMGSZ, opset=OPSET, simplify=True, dynamic=False)
    )
    dest = out_dir / f"{name}.onnx"
    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.move(str(exported), dest)
    return dest


def store_weights_as_fp16(path: Path) -> None:
    """Rewrite large float32 initializers as float16 + Cast(float32) nodes."""
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    model = onnx.load(str(path))
    graph = model.graph
    cast_nodes = []
    new_initializers = []
    for init in list(graph.initializer):
        if init.data_type != TensorProto.FLOAT:
            continue
        arr = numpy_helper.to_array(init)
        if arr.size < MIN_FP16_ELEMENTS:
            continue
        half_name = f"{init.name}__fp16"
        half = numpy_helper.from_array(arr.astype(np.float16), name=half_name)
        new_initializers.append((init, half))
        cast_nodes.append(
            helper.make_node("Cast", [half_name], [init.name], to=TensorProto.FLOAT, name=f"cast_{half_name}")
        )

    for old, half in new_initializers:
        graph.initializer.remove(old)
        graph.initializer.append(half)
    # Casts only depend on initializers, so they can safely come first.
    existing = list(graph.node)
    del graph.node[:]
    graph.node.extend(cast_nodes + existing)
    onnx.checker.check_model(model)
    onnx.save(model, str(path))


def verify(path: Path, reference: Path | None) -> None:
    """Run the exported model and check shapes (and agreement with a float32 reference)."""
    import onnxruntime as ort

    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    x = np.random.default_rng(0).random((1, 3, IMGSZ, IMGSZ), dtype=np.float32)
    outs = sess.run(None, {sess.get_inputs()[0].name: x})
    shapes = [tuple(o.shape) for o in outs]
    assert shapes[0][1] == 4 + 80 + 32 and shapes[1][1:] == (32, 160, 160), shapes
    print(f"  outputs {shapes}")
    if reference is not None:
        ref = ort.InferenceSession(str(reference), providers=["CPUExecutionProvider"])
        ref_outs = ref.run(None, {ref.get_inputs()[0].name: x})
        for a, b in zip(outs, ref_outs):
            print(f"  max |diff| vs float32 export: {np.abs(a - b).max():.4g}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="+", default=["yolov8n-seg", "yolov8s-seg"])
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--fp16-weights", action="store_true")
    args = parser.parse_args()

    for name in args.models:
        print(f"== {name}")
        path = export_onnx(name, args.out)
        size = path.stat().st_size / 1e6
        print(f"  exported {path} ({size:.1f} MB)")
        reference = None
        if args.fp16_weights:
            reference = path.with_suffix(".fp32.onnx")
            shutil.copy(path, reference)
            store_weights_as_fp16(path)
            print(f"  fp16 weights: {path.stat().st_size / 1e6:.1f} MB")
        verify(path, reference)
        if reference is not None:
            reference.unlink()
    return 0


if __name__ == "__main__":
    sys.exit(main())
