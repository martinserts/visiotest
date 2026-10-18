//! The latency table's rows, as data: one list that the page's grid, its colour
//! legend and the Markdown copy all read, so the three can't drift apart.
//!
//! A row names its index in `DashText::latency` (where its formatted cells live), a
//! short name that keeps the p95 column in view, and a hover text that says exactly
//! which clock it reads, from where to where.

use crate::ui::{DETECT_TOTAL, EVENTS, INFER, POST, PRE, SENSOR, TOTAL, TRACK, UPLOAD, WAIT};

/// One row of the table: its index in `DashText::latency`, the short name, the hover
/// text that says exactly what is measured, and whether it is a total (bold).
pub(super) struct Row {
    pub(super) index: usize,
    pub(super) name: &'static str,
    pub(super) hover: &'static str,
    pub(super) total: bool,
}

/// The video path's rows: camera → screen, measured per shown frame.
pub(super) const VIDEO_ROWS: [Row; 4] = [
    Row {
        index: SENSOR,
        name: "Sensor → app",
        hover: "Sensor timestamp → the capture callback: exposure, driver, AVFoundation.",
        total: false,
    },
    Row {
        index: WAIT,
        name: "Wait for UI",
        hover: "Capture callback → the redraw that took the frame.",
        total: false,
    },
    Row {
        index: UPLOAD,
        name: "Upload",
        hover: "CPU time of write_texture + the YUYV → RGB pass.",
        total: false,
    },
    Row {
        index: TOTAL,
        name: "App total",
        hover: "Sum of the stages above, per frame.",
        total: true,
    },
];

/// The detector worker's rows, measured per detector run.
pub(super) const DETECT_ROWS: [Row; 6] = [
    Row {
        index: PRE,
        name: "Preprocess",
        hover: "YUYV → 640×640 tensor (letterbox, normalise).",
        total: false,
    },
    Row {
        index: INFER,
        name: "Inference",
        hover: "ONNX Runtime's session.run only.",
        total: false,
    },
    Row {
        index: POST,
        name: "Postprocess",
        hover: "Decode, NMS, un-letterbox.",
        total: false,
    },
    Row {
        index: DETECT_TOTAL,
        name: "Frame → boxes",
        hover: "Frame arrival → boxes, tracks and events ready (includes the wait in the slot).",
        total: true,
    },
    Row {
        index: TRACK,
        name: "Tracking",
        hover: "Tracker::update + views.",
        total: false,
    },
    Row {
        index: EVENTS,
        name: "Events",
        hover: "Rule engine: sync rules, update, totals.",
        total: false,
    },
];

/// A row's name as shown. Without a sensor timestamp the App total leaves Sensor → app
/// out, and the `*` points to the footnote that says so.
pub(super) fn row_name(row: &Row, partial: bool) -> &'static str {
    if row.index == TOTAL && partial {
        "App total*"
    } else {
        row.name
    }
}
