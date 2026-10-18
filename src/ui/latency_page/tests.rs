//! Tests for the Latency page's pure parts: the waterfall's segments, the Markdown
//! table and Hold. The drawing itself isn't unit-tested.

use super::markdown::markdown;
use super::waterfall::{Segment, sum_ms, waterfall};
use super::*;
use crate::stats::Summary;
use crate::ui::{DETECT_TOTAL, EVENTS, INFER, POST, PRE, SENSOR, TRACK, UPLOAD, WAIT};

/// A summary whose p50 is `p50` (last and p95 don't matter to the waterfall).
fn p50(p50: f32) -> Option<Summary> {
    Some(Summary {
        last: p50 + 1.0,
        p50,
        p95: p50 + 2.0,
    })
}

fn seg(index: usize, ms: f32) -> Segment {
    Segment { index, ms }
}

/// A snapshot with every stage measured: video 75 + 4 + 1 = 80 ms, and a Frame → boxes
/// of 20 ms of which 2 + 12 + 1 + 0.5 + 0.5 = 16 ms are measured stages.
fn full_snapshot() -> TelemetrySnapshot {
    TelemetrySnapshot {
        sensor: p50(75.0),
        wait: p50(4.0),
        upload: p50(1.0),
        total: p50(80.0),
        preprocess: p50(2.0),
        inference: p50(12.0),
        postprocess: p50(1.0),
        tracking: p50(0.5),
        events: p50(0.5),
        detect_total: p50(20.0),
        ..TelemetrySnapshot::default()
    }
}

#[test]
fn waterfall_video_bar_is_the_three_video_p50s() {
    let w = waterfall(&full_snapshot());
    assert_eq!(
        w.video,
        vec![seg(SENSOR, 75.0), seg(WAIT, 4.0), seg(UPLOAD, 1.0)]
    );
}

#[test]
fn waterfall_skips_a_stage_without_data() {
    let s = TelemetrySnapshot {
        sensor: None,
        ..full_snapshot()
    };
    assert_eq!(waterfall(&s).video, vec![seg(WAIT, 4.0), seg(UPLOAD, 1.0)]);
}

#[test]
fn waterfall_boxes_bar_starts_with_the_wait_in_the_slot() {
    let w = waterfall(&full_snapshot());
    assert_eq!(
        w.boxes,
        vec![
            seg(DETECT_TOTAL, 4.0), // 20 − 16 ms measured
            seg(PRE, 2.0),
            seg(INFER, 12.0),
            seg(POST, 1.0),
            seg(TRACK, 0.5),
            seg(EVENTS, 0.5),
        ]
    );
    assert_eq!(sum_ms(&w.boxes), 20.0);
}

#[test]
fn waterfall_wait_in_slot_is_clamped_at_zero() {
    // The stages' p50s add up to 16 ms, more than this Frame → boxes p50.
    let s = TelemetrySnapshot {
        detect_total: p50(15.0),
        ..full_snapshot()
    };
    assert_eq!(waterfall(&s).boxes[0], seg(DETECT_TOTAL, 0.0));
}

#[test]
fn waterfall_has_no_boxes_bar_without_a_detector_result() {
    let s = TelemetrySnapshot {
        detect_total: None,
        ..full_snapshot()
    };
    assert!(waterfall(&s).boxes.is_empty());
}

#[test]
fn waterfall_scale_is_the_longer_bar() {
    assert_eq!(waterfall(&full_snapshot()).scale_ms, 80.0);
    let slow_detector = TelemetrySnapshot {
        detect_total: p50(120.0),
        ..full_snapshot()
    };
    assert_eq!(waterfall(&slow_detector).scale_ms, 120.0);
}

#[test]
fn waterfall_scale_is_at_least_one_ms_on_an_empty_snapshot() {
    let w = waterfall(&TelemetrySnapshot::default());
    assert!(w.video.is_empty() && w.boxes.is_empty());
    assert_eq!(w.scale_ms, 1.0);
}

/// Table cells as `DashText` pads them, with every row set to `last, p50, p95`.
fn padded_cells() -> [[String; 3]; 10] {
    std::array::from_fn(|i| {
        let base = 10.0 * i as f32 + 0.5;
        [base, base + 1.0, base + 2.0].map(|v| format!("{v:>7.1}"))
    })
}

#[test]
fn markdown_header_names_the_ep_and_the_interval() {
    let md = markdown(&padded_cells(), Some("EP CoreML"), "every frame", false);
    assert!(md.starts_with("EP CoreML · detect every frame\n\n"));
}

#[test]
fn markdown_header_without_a_detector() {
    let md = markdown(&padded_cells(), None, "—", false);
    assert!(md.starts_with("detect off\n\n"));
}

#[test]
fn markdown_has_the_table_head_and_alignment_row() {
    let md = markdown(&padded_cells(), Some("EP CPU"), "250 ms", false);
    assert!(md.contains("\n| Stage | last | p50 | p95 |\n|---|--:|--:|--:|\n"));
}

#[test]
fn markdown_trims_padded_cells() {
    let mut cells = padded_cells();
    cells[UPLOAD] = ["   20.5", "   20.0", "   21.0"].map(String::from);
    let md = markdown(&cells, Some("EP CPU"), "every frame", false);
    assert!(md.contains("\n| Upload | 20.5 | 20.0 | 21.0 |\n"));
}

#[test]
fn markdown_keeps_the_no_data_dash() {
    let mut cells = padded_cells();
    cells[SENSOR] = ["      —", "      —", "      —"].map(String::from);
    let md = markdown(&cells, Some("EP CPU"), "every frame", false);
    assert!(md.contains("\n| Sensor → app | — | — | — |\n"));
}

#[test]
fn markdown_makes_totals_bold() {
    let md = markdown(&padded_cells(), Some("EP CPU"), "every frame", false);
    assert!(md.contains("\n| **App total** | **30.5** | **31.5** | **32.5** |\n"));
    assert!(md.contains("\n| **Frame → boxes** | **70.5** | **71.5** | **72.5** |\n"));
}

#[test]
fn markdown_marks_a_partial_total() {
    let md = markdown(&padded_cells(), Some("EP CPU"), "every frame", true);
    assert!(md.contains("| **App total*** |"));
}

#[test]
fn markdown_lists_every_row_with_a_detector() {
    let md = markdown(&padded_cells(), Some("EP CPU"), "every frame", false);
    // Header line, blank line, table head and alignment row, then the 10 rows.
    assert_eq!(md.lines().count(), 4 + 10);
    assert!(md.ends_with("| Events | 90.5 | 91.5 | 92.5 |\n"));
}

#[test]
fn markdown_has_no_detector_rows_without_a_detector() {
    let md = markdown(&padded_cells(), None, "—", false);
    assert_eq!(md.lines().count(), 4 + 4);
    assert!(!md.contains("Inference"));
    assert!(!md.contains("Frame → boxes"));
}

#[test]
fn refresh_builds_the_bars_from_the_snapshot() {
    let mut page = LatencyUi::default();
    page.refresh(&full_snapshot());
    assert_eq!(page.bars, waterfall(&full_snapshot()));
    assert_eq!(page.bar_totals, ["80.0 ms", "20.0 ms"].map(String::from));
}

#[test]
fn hold_keeps_the_bars_and_the_table_through_refresh() {
    let mut page = LatencyUi::default();
    let mut text = DashText::default();
    text.latency[SENSOR][1] = "   75.0".to_string();
    page.hold(&full_snapshot(), &text);

    let later = TelemetrySnapshot {
        sensor: p50(90.0),
        ..full_snapshot()
    };
    page.refresh(&later);

    assert!(page.is_held());
    assert_eq!(page.bars, waterfall(&full_snapshot()));
    let held = page.held.as_ref().expect("held");
    assert_eq!(held.snap, full_snapshot());
    assert_eq!(held.cells[SENSOR][1], "   75.0");
}

#[test]
fn release_clears_hold_and_the_next_refresh_catches_up() {
    let mut page = LatencyUi::default();
    page.hold(&full_snapshot(), &DashText::default());
    page.release();
    assert!(!page.is_held());

    let later = TelemetrySnapshot {
        sensor: p50(90.0),
        ..full_snapshot()
    };
    page.refresh(&later);
    assert_eq!(page.bars, waterfall(&later));
}

#[test]
fn copy_uses_the_held_numbers_while_held() {
    let mut page = LatencyUi::default();
    let mut text = DashText::default();
    text.latency = padded_cells();
    page.hold(&full_snapshot(), &text);
    text.latency[UPLOAD] = ["   99.0", "   99.0", "   99.0"].map(String::from);

    let md = page.copy_text(&full_snapshot(), &text, Some("EP CPU"));
    assert!(md.contains("| Upload | 20.5 | 21.5 | 22.5 |"));
}
