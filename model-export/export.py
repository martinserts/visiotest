"""Export Ultralytics' pretrained YOLO11n to ONNX and verify the result.

Usage (from this directory):

    uv run export.py [--weights yolo11n.pt] [--out ../models/yolo11n.onnx]

The Rust side (visiotest, in the repository root) relies on this contract:

    input   images   float32[1, 3, 640, 640]   RGB, NCHW, values in [0, 1]
    output  output0  float32[1, 84, 8400]

What 84 x 8400 means
--------------------
YOLO11 predicts at three scales. With a 640x640 input and strides 8, 16 and 32, the
grids are 80x80, 40x40 and 20x20 cells: 6400 + 1600 + 400 = 8400 "anchors" (one
prediction per grid cell). Each anchor has 84 attributes:

    0..4    cx, cy, w, h   box centre and size, in 640x640 input pixels
    4..84   80 class scores, already passed through a sigmoid (0..1)

YOLO11 has no separate objectness score: an anchor's confidence is its best class score.

The tensor is attribute-major (84 rows of 8400 values), so in the flat buffer
`out[a * 8400 + i]` is attribute `a` of anchor `i`.

Why NMS is not in the model
---------------------------
Many anchors fire on the same object. Non-maximum suppression keeps the best box and
drops its overlapping neighbours. We export with `nms=False` and write NMS by hand in
Rust, because understanding it is part of the point of this project.
"""

import argparse
import ast
import os
import shutil
from pathlib import Path

# Ultralytics lists onnxruntime as an export requirement when `simplify=True` and tries
# to pip-install it if missing. We never run inference in Python (that's Rust's job), the
# export works without it, and a pip call inside a uv venv fails anyway. Turn
# auto-install off; this must be set before `ultralytics` is imported.
os.environ["YOLO_AUTOINSTALL"] = "false"

import onnx  # noqa: E402
from ultralytics import YOLO  # noqa: E402

# The input size is fixed: visiotest preprocesses to 640x640, and 8400 below follows from it.
IMGSZ = 640
EXPECTED_INPUT = ("images", [1, 3, IMGSZ, IMGSZ])
EXPECTED_OUTPUT = ("output0", [1, 84, 8400])
NUM_CLASSES = 80


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export YOLO11n to ONNX for visiotest.")
    parser.add_argument("--weights", default="yolo11n.pt", help="Ultralytics weights")
    parser.add_argument("--out", default="../models/yolo11n.onnx", help="where to write the model")
    return parser.parse_args()


def export(weights: str, out: Path) -> None:
    """Export `weights` to ONNX and move the file to `out` (overwriting it)."""
    # If `weights` is a known model name and the file is missing, Ultralytics downloads it.
    model = YOLO(weights)
    exported = model.export(
        format="onnx",
        imgsz=IMGSZ,
        # Static shapes: every dimension is a fixed number, so ONNX Runtime and CoreML
        # can plan (and CoreML compile) the whole graph once.
        dynamic=False,
        # Fold constants and remove redundant nodes with onnxslim. Same maths, smaller graph.
        simplify=True,
        # No NMS inside the graph: we write NMS ourselves in Rust.
        nms=False,
        # FP32 weights; FP16/INT8 variants are not exported yet. Ultralytics 8.4
        # replaced the old `half=`/`int8=` flags with `quantize=`.
        quantize=32,
        # Let Ultralytics choose the newest opset its torch exporter supports.
        opset=None,
    )
    # Ultralytics writes `<weights stem>.onnx` next to the weights, not where we want it.
    out.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(exported), str(out))


def tensor_info(value: onnx.ValueInfoProto) -> tuple[str, str, list[int]]:
    """Return (name, element type, dims) of a graph input or output."""
    tensor = value.type.tensor_type
    elem = onnx.TensorProto.DataType.Name(tensor.elem_type)
    dims = [d.dim_value if d.HasField("dim_value") else d.dim_param for d in tensor.shape.dim]
    return value.name, elem, dims


def check_io(kind: str, values, expected: tuple[str, list[int]]) -> None:
    """Print the single input or output and fail unless it matches the contract."""
    assert len(values) == 1, f"expected one {kind}, found {len(values)}"
    name, elem, dims = tensor_info(values[0])
    print(f"{kind:<6} {name}: {elem}{dims}")
    assert elem == "FLOAT", f"{kind} {name!r} is {elem}, expected FLOAT (float32)"
    assert (name, dims) == expected, (
        f"{kind} is {name!r} {dims}, expected {expected[0]!r} {expected[1]}"
    )


def check_model(path: Path) -> list[str]:
    """Validate the ONNX file, check its I/O contract and return the class names."""
    model = onnx.load(str(path))
    onnx.checker.check_model(model)

    check_io("input", model.graph.input, EXPECTED_INPUT)
    check_io("output", model.graph.output, EXPECTED_OUTPUT)

    # Ultralytics stores its metadata as strings. `names` is a Python dict literal,
    # e.g. "{0: 'person', 1: 'bicycle', ...}", so parse it safely with literal_eval.
    metadata = {p.key: p.value for p in model.metadata_props}
    names = ast.literal_eval(metadata["names"])
    assert len(names) == NUM_CLASSES, f"expected {NUM_CLASSES} classes, found {len(names)}"
    return [names[i] for i in range(NUM_CLASSES)]


def main() -> None:
    args = parse_args()
    out = Path(args.out)
    export(args.weights, out)

    print()
    print(f"verifying {out}")
    names = check_model(out)
    print(f"classes ({len(names)}, YOLO/COCO order):")
    for i, name in enumerate(names):
        print(f"{i:>2}: {name}")

    size_mb = out.stat().st_size / 1_000_000
    print(f"wrote {out} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
