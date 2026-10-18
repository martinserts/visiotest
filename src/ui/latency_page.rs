//! The Latency page: the one home of the full per-stage latency table.
//!
//! Every stage's timing is measured where it happens (the capture callback, the UI
//! thread, the detector worker) and summarised as last / p50 / p95 over the last few
//! seconds. Stage pages never repeat these rows, so a number can't disagree with itself.
//!
//! The page has three parts:
//!
//! - a **waterfall** (`waterfall.rs`): the p50s as two stacked bars on one ms scale,
//!   Video (sensor → screen) and Boxes (frame → boxes), so the biggest block shows at a
//!   glance. Each row's swatch in the table is its block's colour.
//! - the **table**, built from one row list (`rows.rs`: `VIDEO_ROWS`, `DETECT_ROWS`):
//!   short names, with the exact clock each row reads in its hover text.
//! - a toolbar: **Hold** freezes the table and the bars so they can be read or
//!   screenshotted calmly (the strip keeps running), and **Copy** puts the table on the
//!   clipboard as Markdown (`markdown.rs`).
//!
//! A new stage adds its row the same way: push the stage's timing into `Telemetry`,
//! carry its `Summary` in the snapshot, and add a `Row` in `rows.rs`.

mod markdown;
mod rows;
#[cfg(test)]
mod tests;
mod waterfall;

use std::fmt::Write;

use eframe::egui::{self, Color32, Label, Painter, Rect, RichText, Sense, vec2};

use super::{DashText, metric_style, section};
use crate::telemetry::TelemetrySnapshot;
use markdown::markdown;
use rows::{DETECT_ROWS, Row, VIDEO_ROWS, row_name};
use waterfall::{Segment, Waterfall, sum_ms, waterfall};

/// Header cells, padded to the number-cell width.
const HEADER: [&str; 3] = ["   last", "    p50", "    p95"];

/// Each row's block colour, indexed like `DashText::latency`: cool for the video path,
/// warm for the detector, so the two bars read apart. None is a box colour
/// (`detections::GROUP_COLORS`): a block means "a stage", not "a class".
const SEGMENT_COLORS: [Color32; 10] = [
    Color32::from_rgb(0x5b, 0x8d, 0xef), // SENSOR: blue
    Color32::from_rgb(0x8e, 0x7c, 0xf0), // WAIT: violet
    Color32::from_rgb(0x3f, 0xc1, 0xc9), // UPLOAD: teal
    Color32::TRANSPARENT,                // TOTAL: a sum, so no block of its own
    Color32::from_rgb(0xf2, 0xc1, 0x4e), // PRE: yellow
    Color32::from_rgb(0xf2, 0x8c, 0x38), // INFER: orange
    Color32::from_rgb(0xe4, 0x57, 0x2e), // POST: red-orange
    // DETECT_TOTAL's block is the wait in the slot: grey, because it is time nobody
    // works on the frame, not a stage.
    Color32::from_gray(0x6a),
    Color32::from_rgb(0xd9, 0x48, 0x7d), // TRACK: rose
    Color32::from_rgb(0xb0, 0x6e, 0x3a), // EVENTS: brown
];

/// Height of one waterfall bar, in points.
const BAR_H: f32 = 14.0;
/// Width of the bar's left label ("Video", "Boxes"), so both bars start at one x.
const BAR_LABEL_W: f32 = 40.0;
/// Room for the bar's total on its right (`123.4 ms`).
const BAR_TOTAL_W: f32 = 64.0;
/// A block is at least this wide, so a stage of a fraction of a ms never silently
/// disappears from its bar.
const MIN_SEGMENT_PX: f32 = 1.0;
/// Side of a table row's colour swatch.
const SWATCH: f32 = 10.0;

const VIDEO_HOVER: &str = "p50s: Sensor → app, Wait for UI, Upload.";
const BOXES_HOVER: &str = "p50s: the wait in the detector's slot (grey: Frame → boxes minus \
     the measured stages), then Preprocess … Events. Both bars share one ms scale.";

/// What Hold froze: the snapshot and the table's strings at the moment it was pressed.
struct Held {
    snap: TelemetrySnapshot,
    cells: [[String; 3]; 10],
    /// The detect interval at that moment, for Copy's header line.
    interval: String,
}

/// The page's own state: what Hold froze, and the bars to draw.
#[derive(Default)]
pub struct LatencyUi {
    /// `Some` while Hold is on.
    held: Option<Held>,
    /// Rebuilt by `refresh` when the snapshot changes (unless held).
    bars: Waterfall,
    /// Each bar's total, `80.1 ms` (Video, Boxes), formatted with the bars.
    bar_totals: [String; 2],
    /// The snapshot the bars were built from.
    last: Option<TelemetrySnapshot>,
}

impl LatencyUi {
    /// Rebuilds the bars when the snapshot changed, unless Hold is on. Called on every
    /// redraw: `App` rebuilds the snapshot only on its 4 Hz tick, which the panel doesn't
    /// see, so comparing with the last snapshot finds the tick. A redraw then costs one
    /// `==` on a small `Copy` struct, and the bars are rebuilt at 4 Hz.
    pub fn refresh(&mut self, s: &TelemetrySnapshot) {
        if self.is_held() || self.last == Some(*s) {
            return;
        }
        self.rebuild(s);
    }

    /// Whether Hold is on.
    pub fn is_held(&self) -> bool {
        self.held.is_some()
    }

    /// Turns Hold on: freezes this snapshot, the table's strings and the bars.
    pub fn hold(&mut self, s: &TelemetrySnapshot, text: &DashText) {
        // Rebuilt from `s` itself, so the bars always match the frozen table.
        self.rebuild(s);
        self.held = Some(Held {
            snap: *s,
            cells: text.latency.clone(),
            interval: text.detect_interval.clone(),
        });
    }

    /// Turns Hold off. The next `refresh` with a newer snapshot rebuilds the bars.
    pub fn release(&mut self) {
        self.held = None;
    }

    /// The Markdown that Copy puts on the clipboard: the held numbers while held, else
    /// the live ones.
    fn copy_text(&self, snap: &TelemetrySnapshot, text: &DashText, ep: Option<&str>) -> String {
        match &self.held {
            Some(h) => markdown(&h.cells, ep, &h.interval, h.snap.total_partial),
            None => markdown(&text.latency, ep, &text.detect_interval, snap.total_partial),
        }
    }

    fn rebuild(&mut self, s: &TelemetrySnapshot) {
        self.last = Some(*s);
        self.bars = waterfall(s);
        write_total(&mut self.bar_totals[0], &self.bars.video);
        write_total(&mut self.bar_totals[1], &self.bars.boxes);
    }
}

/// `80.1 ms` into a reused string, or `—` for a bar with no data yet.
fn write_total(out: &mut String, bar: &[Segment]) {
    out.clear();
    if bar.is_empty() {
        out.push('—');
    } else {
        // Writing into a String can't fail.
        let _ = write!(out, "{:.1} ms", sum_ms(bar));
    }
}

/// The page: the toolbar, **Waterfall**, **Table** and its footnotes. `ep` is the
/// status bar's `EP CoreML` chip (`None` with `--no-detect`), for Copy's header line.
pub fn show(
    ui: &mut egui::Ui,
    page: &mut LatencyUi,
    snap: &TelemetrySnapshot,
    text: &DashText,
    ep: Option<&str>,
) {
    toolbar(ui, page, snap, text, ep);
    let (snap, cells) = match &page.held {
        Some(h) => (&h.snap, &h.cells),
        None => (snap, &text.latency),
    };
    // `section` brings its own space above, so no extra gaps between the parts.
    section(ui, "Waterfall");
    waterfall_bars(ui, &page.bars, &page.bar_totals, text.detection());
    section(ui, "Table");
    table(ui, cells, snap.total_partial, text.detection());
    footnotes(ui, snap, text);
}

/// Hold and Copy, and a reminder while held that only this page is frozen.
fn toolbar(
    ui: &mut egui::Ui,
    page: &mut LatencyUi,
    snap: &TelemetrySnapshot,
    text: &DashText,
    ep: Option<&str>,
) {
    ui.horizontal(|ui| {
        let mut held = page.is_held();
        if ui.toggle_value(&mut held, "⏸ Hold").changed() {
            if held {
                page.hold(snap, text);
            } else {
                page.release();
            }
        }
        let copy = ui
            .button("Copy")
            .on_hover_text("Copy the table as Markdown.");
        if copy.clicked() {
            // Formatted only on the click, never per redraw.
            ui.ctx().copy_text(page.copy_text(snap, text, ep));
        }
    });
    if page.is_held() {
        ui.label(
            RichText::new("Held — the strip is still live.")
                .small()
                .weak(),
        );
    }
}

/// The two bars, Boxes only with a detector.
fn waterfall_bars(ui: &mut egui::Ui, bars: &Waterfall, totals: &[String; 2], detection: bool) {
    bar_row(
        ui,
        "Video",
        VIDEO_HOVER,
        &bars.video,
        bars.scale_ms,
        &totals[0],
    );
    if detection {
        bar_row(
            ui,
            "Boxes",
            BOXES_HOVER,
            &bars.boxes,
            bars.scale_ms,
            &totals[1],
        );
    }
}

/// One bar: its name on the left, the blocks, its total on the right.
fn bar_row(
    ui: &mut egui::Ui,
    name: &str,
    hover: &str,
    segments: &[Segment],
    scale_ms: f32,
    total: &str,
) {
    ui.horizontal(|ui| {
        ui.add_sized(
            [BAR_LABEL_W, BAR_H],
            Label::new(RichText::new(name).small()),
        )
        .on_hover_text(hover);
        let width = (ui.available_width() - BAR_TOTAL_W).max(0.0);
        let (rect, response) = ui.allocate_exact_size(vec2(width, BAR_H), Sense::hover());
        let track = ui.visuals().extreme_bg_color;
        paint_bar(ui.painter(), rect, segments, scale_ms, track);
        response.on_hover_text(hover);
        ui.label(RichText::new(total).small());
    });
}

/// Lays the blocks left to right, each `width × ms / scale_ms` wide, over a dark track
/// that shows the shared scale (a shorter bar leaves part of its track empty).
fn paint_bar(painter: &Painter, rect: Rect, segments: &[Segment], scale_ms: f32, track: Color32) {
    painter.rect_filled(rect, 2.0, track);
    let mut x = rect.left();
    for seg in segments {
        let width = (rect.width() * seg.ms / scale_ms).max(MIN_SEGMENT_PX);
        // The 1 px minimum can push the last blocks past the end; clip them there.
        let right = (x + width).min(rect.right());
        if right <= x {
            break;
        }
        let block = Rect::from_x_y_ranges(x..=right, rect.y_range());
        painter.rect_filled(block, 0.0, SEGMENT_COLORS[seg.index]);
        x = right;
    }
}

/// The grid: video rows, the detector sub-header, the detector rows.
fn table(ui: &mut egui::Ui, cells: &[[String; 3]; 10], partial: bool, detection: bool) {
    // The page shell draws the "Latency" heading, so the unit goes in the header cell.
    egui::Grid::new("latency_grid")
        .num_columns(4)
        .striped(true)
        .spacing([12.0, 4.0])
        .show(ui, |ui| {
            ui.strong("Stage (ms)");
            for h in HEADER {
                ui.label(RichText::new(h).text_style(metric_style()).strong());
            }
            ui.end_row();

            for row in &VIDEO_ROWS {
                measured_row(ui, row, &cells[row.index], partial);
            }
            let worker = if detection {
                "Detection (worker)"
            } else {
                "Detection (off)"
            };
            sub_header(ui, worker);
            for row in &DETECT_ROWS {
                measured_row(ui, row, &cells[row.index], partial);
            }
        });
}

/// The weak notes under the table.
fn footnotes(ui: &mut egui::Ui, snap: &TelemetrySnapshot, text: &DashText) {
    if snap.total_partial {
        ui.label(
            RichText::new("* no sensor timestamp: the total leaves out Sensor → app.")
                .small()
                .weak(),
        );
    }
    if text.detection() {
        ui.label(
            RichText::new(
                "Tracks are predicted to the frame on screen; raw detections \
                 trail it by Boxes behind frames.",
            )
            .small()
            .weak(),
        );
    }
    ui.add_space(4.0);
    ui.label(
        RichText::new(
            "Glass-to-glass: point the camera at this window and compare the clock \
             with its image in the video. Toggle with C.",
        )
        .small()
        .weak(),
    );
}

/// A small, weak label spanning the first column, grouping the rows below it.
fn sub_header(ui: &mut egui::Ui, text: &str) {
    ui.label(RichText::new(text).small().weak());
    ui.end_row();
}

/// One table row: swatch and name (hover: what is measured), then last / p50 / p95.
fn measured_row(ui: &mut egui::Ui, row: &Row, cells: &[String; 3], partial: bool) {
    ui.horizontal(|ui| {
        swatch(ui, SEGMENT_COLORS[row.index]);
        let mut name = RichText::new(row_name(row, partial));
        if row.total {
            name = name.strong();
        }
        ui.label(name).on_hover_text(row.hover);
    });
    for c in cells {
        let mut v = RichText::new(c.as_str()).text_style(metric_style());
        if row.total {
            v = v.strong();
        }
        ui.label(v);
    }
    ui.end_row();
}

/// The row's legend square, in its waterfall block's colour. App total's is
/// transparent, so its name still lines up with the others.
fn swatch(ui: &mut egui::Ui, color: Color32) {
    let (rect, _) = ui.allocate_exact_size(vec2(SWATCH, SWATCH), Sense::hover());
    ui.painter().rect_filled(rect, 2.0, color);
}
