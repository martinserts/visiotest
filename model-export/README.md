# model-export

Exports Ultralytics' pretrained **YOLO11n** to ONNX for visiotest. Python is used
only to export and convert models; everything that runs is Rust.

## Export

```sh
cd model-export && uv run export.py
```

The first run installs torch and downloads `yolo11n.pt` (several hundred MB in total).
Options: `--weights yolo11n.pt`, `--out ../models/yolo11n.onnx`. Running it again
overwrites the output.

The model lands in `models/yolo11n.onnx` at the repository root, which is visiotest's
default `--model`, so `cargo run --release` from the root works with no flags.
`models/` and `*.pt` are gitignored, so neither the model nor the weights are committed.

## Contract

The script checks the exported file and fails unless it has:

| | name | type | shape |
|---|---|---|---|
| input | `images` | float32 | `[1, 3, 640, 640]` (RGB, NCHW, 0..1) |
| output | `output0` | float32 | `[1, 84, 8400]` |

`8400` = 80x80 + 40x40 + 20x20 grid cells (strides 8/16/32). `84` = `cx, cy, w, h` in
input pixels, then 80 sigmoid class scores (no objectness score). The layout is
attribute-major: `out[a * 8400 + i]` is attribute `a` of anchor `i`. The export is
static-shape, FP32, and has **no NMS** in the graph (visiotest does NMS by hand). The
script also prints the 80 class names in order, which is the reference for the Rust
`CLASS_NAMES` table.

## License

This script is MIT OR Apache-2.0, like the rest of the repository. It imports
[Ultralytics](https://github.com/ultralytics/ultralytics), which is **AGPL-3.0**, and the
YOLO11 weights and the exported model are covered by Ultralytics' license too. That is
why neither is committed here. If AGPL doesn't suit your use, RF-DETR and D-FINE are
Apache-2.0 detectors.
