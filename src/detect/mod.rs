//! Object detection: everything around the model is written by hand.
//!
//! ```text
//! YUYV frame ─► [pace?] interval: run on this frame, or skip it?
//!            ─► Preprocessor ─► 1×3×640×640 f32 tensor
//!            ─► Model (YOLO11n in ONNX Runtime) ─► [84 × 8400] f32 output
//!            ─► Postprocessor ─► Vec<Detection> in frame pixels
//!            ─► Tracker (crate::track) ─► Detections + tracks
//! ```
//!
//! [`worker`] runs that pipeline on its own thread; [`interval`] decides how often. Only [`model`] touches ONNX
//! Runtime; the rest never touches the camera, the GPU, a window or a model file, so it
//! is unit-tested with synthetic frames and synthetic output tensors.

pub mod classes;
pub mod interval;
pub mod model;
pub mod postprocess;
pub mod preprocess;
pub mod worker;

/// Side of the square model input. The YOLO11n export is fixed at 1×3×640×640.
pub const INPUT_SIZE: u32 = 640;
/// COCO has 80 object classes.
pub const NUM_CLASSES: usize = 80;
/// One prediction per grid cell over three scales: 80² + 40² + 20² cells
/// (strides 8, 16 and 32 on a 640×640 input).
pub const NUM_ANCHORS: usize = 8400;
/// Upper bound on boxes per frame, the same default Ultralytics uses (`max_det=300`).
pub const MAX_DETECTIONS: usize = 300;

/// Axis-aligned box in **frame pixels** (x0,y0 top-left; x1,y1 bottom-right).
///
/// Inside the postprocessor the same type also holds boxes in 640×640 input space,
/// before they are un-letterboxed.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BBox {
    pub x0: f32,
    pub y0: f32,
    pub x1: f32,
    pub y1: f32,
}

impl BBox {
    /// Builds a box from its top-left `(x0, y0)` and bottom-right `(x1, y1)` corners.
    #[allow(dead_code)] // only tests build boxes from corners so far
    pub fn new(x0: f32, y0: f32, x1: f32, y1: f32) -> Self {
        Self { x0, y0, x1, y1 }
    }

    /// Builds a box from its centre and size, the form YOLO predicts boxes in.
    pub fn from_center(cx: f32, cy: f32, w: f32, h: f32) -> Self {
        Self {
            x0: cx - w / 2.0,
            y0: cy - h / 2.0,
            x1: cx + w / 2.0,
            y1: cy + h / 2.0,
        }
    }

    pub fn width(&self) -> f32 {
        self.x1 - self.x0
    }

    pub fn height(&self) -> f32 {
        self.y1 - self.y0
    }

    /// Area, or 0 for an empty or inverted box (so it never goes negative).
    pub fn area(&self) -> f32 {
        self.width().max(0.0) * self.height().max(0.0)
    }
}

/// One detected object: where it is, what it is, and how sure the model is.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Detection {
    pub bbox: BBox,
    /// COCO class id, an index into [`classes::CLASS_NAMES`].
    pub class: u8,
    /// The class score in `[0, 1]`.
    pub score: f32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bbox_from_center_and_size() {
        let b = BBox::from_center(10.0, 20.0, 4.0, 6.0);
        assert_eq!(b, BBox::new(8.0, 17.0, 12.0, 23.0));
        assert_eq!((b.width(), b.height(), b.area()), (4.0, 6.0, 24.0));
    }

    #[test]
    fn bbox_inverted_has_zero_area() {
        let b = BBox::new(5.0, 0.0, 1.0, 3.0);
        assert_eq!(b.area(), 0.0);
    }

    #[test]
    fn anchors_are_three_grids() {
        let cells = |stride: usize| (INPUT_SIZE as usize / stride).pow(2);
        assert_eq!(cells(8) + cells(16) + cells(32), NUM_ANCHORS);
    }
}
