"""Exports the YOLOv8 segmentation models to ONNX, for onnxruntime-web.

    pip install -r scripts/requirements.txt
    python scripts/export_models.py --fp16-weights      # writes web/models/*.onnx

--fp16-weights stores the weights as float16 and casts them back to float32 when the
session starts: the file is half the size and the model computes the same thing.
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
OPSET = 12  # well supported by the wasm backend
MIN_FP16_ELEMENTS = 1024  # small tensors stay float32


def export_onnx(name: str, out_dir: Path) -> Path:
    from ultralytics import YOLO

    model = YOLO(f"{name}.pt")  # downloads the weights if needed
    exported = Path(
        model.export(format="onnx", imgsz=IMGSZ, opset=OPSET, simplify=True, dynamic=False)
    )
    dest = out_dir / f"{name}.onnx"
    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.move(str(exported), dest)
    return dest


def store_weights_as_fp16(path: Path) -> None:
    """Big float32 weights become float16 + a Cast back to float32."""
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
    # the casts only read weights, so they can go first
    existing = list(graph.node)
    del graph.node[:]
    graph.node.extend(cast_nodes + existing)
    onnx.checker.check_model(model)
    onnx.save(model, str(path))


def verify(path: Path, reference: Path | None) -> None:
    """Runs the model once and checks the output shapes (and the float32 export, if given)."""
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
