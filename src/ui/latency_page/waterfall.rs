//! The waterfall: the latency table as two stacked bars, so the biggest block shows at
//! a glance.
//!
//! Each bar lays its stages end to end, one coloured **segment** per stage, sized by
//! its p50:
//!
//! ```text
//! Video  [ Sensor → app ......... | Wait | Upload ]
//! Boxes  [ wait in slot | Pre | Inference ..... | Post | Tracking | Events ]
//! ```
//!
//! Both bars share one ms scale (the longer bar's length), so their lengths can be
//! compared. The Boxes bar starts with the time a frame **waited in the detector's
//! slot** before the worker took it. Nobody times that wait directly: it is what is
//! left of Frame → boxes once the measured stages are taken out.
//!
//! Building a [`Waterfall`] allocates two small `Vec`s, so the page builds one only
//! when the snapshot changed (4 Hz), never per redraw.

use crate::telemetry::TelemetrySnapshot;
use crate::ui::{DETECT_TOTAL, EVENTS, INFER, POST, PRE, SENSOR, TRACK, UPLOAD, WAIT};

/// The smallest scale, in ms, so drawing never divides by zero before the first frame.
const MIN_SCALE_MS: f32 = 1.0;

/// One coloured block of a bar: which row it is (an index into `DashText::latency`)
/// and its length in ms. The Boxes bar's wait-in-slot block uses `DETECT_TOTAL`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Segment {
    pub index: usize,
    pub ms: f32,
}

/// The two bars. A bar is empty while its data isn't there yet.
#[derive(Clone, Debug, PartialEq)]
pub struct Waterfall {
    /// Sensor → app, Wait for UI, Upload.
    pub video: Vec<Segment>,
    /// Wait in slot, Preprocess, Inference, Postprocess, Tracking, Events; empty
    /// without a detector result.
    pub boxes: Vec<Segment>,
    /// The ms that a bar's full width stands for: the longer bar's sum, at least 1.0.
    pub scale_ms: f32,
}

impl Default for Waterfall {
    /// No bars yet, on the smallest scale.
    fn default() -> Self {
        Self {
            video: Vec::new(),
            boxes: Vec::new(),
            scale_ms: MIN_SCALE_MS,
        }
    }
}

/// The bars for one snapshot, built from p50s. A stage with no data yet is left out.
pub fn waterfall(s: &TelemetrySnapshot) -> Waterfall {
    let video: Vec<Segment> = [(SENSOR, s.sensor), (WAIT, s.wait), (UPLOAD, s.upload)]
        .into_iter()
        .filter_map(|(index, summary)| summary.map(|m| Segment { index, ms: m.p50 }))
        .collect();
    let boxes = boxes_bar(s);
    let scale_ms = sum_ms(&video).max(sum_ms(&boxes)).max(MIN_SCALE_MS);
    Waterfall {
        video,
        boxes,
        scale_ms,
    }
}

/// The length of a bar: the sum of its segments, in ms.
pub fn sum_ms(bar: &[Segment]) -> f32 {
    bar.iter().map(|seg| seg.ms).sum()
}

/// The Boxes bar: the wait in the slot, then the worker's measured stages.
fn boxes_bar(s: &TelemetrySnapshot) -> Vec<Segment> {
    let Some(total) = s.detect_total else {
        return Vec::new();
    };
    let stages = [
        (PRE, s.preprocess),
        (INFER, s.inference),
        (POST, s.postprocess),
        (TRACK, s.tracking),
        (EVENTS, s.events),
    ];
    let measured: Vec<Segment> = stages
        .into_iter()
        .filter_map(|(index, summary)| summary.map(|m| Segment { index, ms: m.p50 }))
        .collect();
    // Clamped at 0: the p50 of a sum isn't the sum of the p50s (each series has its own
    // middle sample), so the stages' p50s can add up to more than Frame → boxes' p50.
    let wait = (total.p50 - sum_ms(&measured)).max(0.0);
    let mut bar = Vec::with_capacity(measured.len() + 1);
    bar.push(Segment {
        index: DETECT_TOTAL,
        ms: wait,
    });
    bar.extend(measured);
    bar
}
