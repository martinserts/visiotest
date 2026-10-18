//! The panel text cache: every string the dashboard shows, rebuilt at 4 Hz.
//!
//! Numbers are formatted into reused `String`s, pre-padded to [`CELL`] characters so
//! the monospace columns stay right-aligned, and `—` means "no data". The pipeline
//! strip's row values (`camera_strip` … `latency_strip`) live here too, so the strip
//! formats nothing when it draws.

use std::fmt::Write;

use crate::app::SourceInfo;
use crate::events::event::EventTotals;
use crate::telemetry::TelemetrySnapshot;

/// Width of every number cell, so monospace columns line up right-aligned.
pub const CELL: usize = 7;

/// Index of each measured row in [`DashText::latency`], in pipeline order.
pub const SENSOR: usize = 0;
pub const WAIT: usize = 1;
pub const UPLOAD: usize = 2;
pub const TOTAL: usize = 3;
/// The detector worker's rows.
pub const PRE: usize = 4;
pub const INFER: usize = 5;
pub const POST: usize = 6;
/// Frame arrival → boxes, tracks and events ready.
pub const DETECT_TOTAL: usize = 7;
/// `Tracker::update` + `views`, shown with 2 decimals.
pub const TRACK: usize = 8;
/// The rule engine, which runs after tracking; 2 decimals too.
pub const EVENTS: usize = 9;

/// Strings the dashboard shows, rebuilt only on the throttled (4 Hz) refresh.
/// The buffers are reused, so a refresh does not allocate once they have grown.
#[derive(Default)]
pub struct DashText {
    /// Status bar: `1920×1080 · YUYV · 30 fps` (static).
    pub source: String,
    /// Camera grid: `1920×1080 YUYV` (static).
    pub resolution: String,
    pub camera_chip: String,
    pub display_chip: String,
    pub frames: String,
    pub dropped: String,
    pub camera_fps: String,
    pub display_fps: String,
    pub interval: String,
    pub detector_fps: String,
    /// Skipped (busy): detector slot overwrites, plus errors.
    pub det_skipped: String,
    /// Skipped (paced): frames the detect interval said to skip.
    pub det_paced: String,
    pub boxes_behind: String,
    /// `every frame` or `250 ms`.
    pub detect_interval: String,
    /// `3 (+1 lost)`, or `3` when none are lost.
    pub tracks: String,
    /// `7 · 2 recovered`.
    pub track_ids: String,
    /// `[last, p50, p95]` per measured row (`SENSOR` … `EVENTS`).
    pub latency: [[String; 3]; 10],
    /// Strip, Camera row: `29.9 fps`; `—` before the first frame.
    pub camera_strip: String,
    /// Strip, Detect row: `30.0 fps`; `—` without a result, `off` with `--no-detect`.
    pub detect_strip: String,
    /// Strip, Track row: `3 · 1 lost`, or `3` when none are lost; `—` / `off` as Detect.
    pub track_strip: String,
    /// Strip, Events row: `5 counted`, `no rules`, `✏ editing` or `off`.
    pub events_strip: String,
    /// Strip, Latency row: App total and Frame → boxes p50, `80 · 16 ms`; `80 ms`
    /// without a detector result, `—` before the first frame.
    pub latency_strip: String,
    /// Whether a detector runs at all (`--no-detect` turns it off).
    detection: bool,
}

impl DashText {
    pub fn new(info: &SourceInfo, detection: bool) -> Self {
        let mut t = Self {
            detection,
            source: format!(
                "{}×{} · {} · {} fps",
                info.width, info.height, info.format, info.fps
            ),
            resolution: format!("{}×{} {}", info.width, info.height, info.format),
            ..Self::default()
        };
        t.refresh(&TelemetrySnapshot::default());
        t.refresh_events(0, None, false);
        t
    }

    pub fn refresh(&mut self, s: &TelemetrySnapshot) {
        set(
            &mut self.camera_chip,
            format_args!("cam {:.1} fps", s.camera_fps),
        );
        set(
            &mut self.display_chip,
            format_args!("ui {:.0} fps", s.display_fps),
        );
        set(&mut self.frames, format_args!("{}", s.seq));
        if s.bad_buffers > 0 {
            set(
                &mut self.dropped,
                format_args!("{} (+{} bad buffers)", s.dropped, s.bad_buffers),
            );
        } else {
            set(&mut self.dropped, format_args!("{}", s.dropped));
        }
        set(&mut self.camera_fps, format_args!("{:.1}", s.camera_fps));
        set(&mut self.display_fps, format_args!("{:.1}", s.display_fps));
        match s.interval {
            Some(i) => set(
                &mut self.interval,
                format_args!("{:.1} / {:.1} ms", i.last, i.p95),
            ),
            None => set(&mut self.interval, format_args!("—")),
        }
        if s.seq == 0 {
            set(&mut self.camera_strip, format_args!("—"));
        } else {
            set(
                &mut self.camera_strip,
                format_args!("{:.1} fps", s.camera_fps),
            );
        }

        self.refresh_detector(s);
        self.refresh_latency_strip(s);

        let rows = [
            s.sensor,
            s.wait,
            s.upload,
            s.total,
            s.preprocess,
            s.inference,
            s.postprocess,
            s.detect_total,
        ];
        for (cells, summary) in self.latency.iter_mut().zip(rows) {
            let [last, p50, p95] = cells;
            cell(last, summary.map(|v| v.last));
            cell(p50, summary.map(|v| v.p50));
            cell(p95, summary.map(|v| v.p95));
        }
        for (row, summary) in [(TRACK, s.tracking), (EVENTS, s.events)] {
            let [last, p50, p95] = &mut self.latency[row];
            fine_cell(last, summary.map(|v| v.last));
            fine_cell(p50, summary.map(|v| v.p50));
            fine_cell(p95, summary.map(|v| v.p95));
        }
        if s.seq > 0 && !s.has_sensor {
            set(
                &mut self.latency[SENSOR][0],
                format_args!("{:>CELL$}", "n/a"),
            );
        }
    }

    /// The strip's Events row. `rules` is how many rules exist, `totals` the newest
    /// fresh totals (`None` without a result) and `editing` whether edit mode is on.
    /// "Counted" sums every line crossing (both ways) and every zone visit, so one
    /// number says whether anything is happening.
    pub fn refresh_events(&mut self, rules: usize, totals: Option<&EventTotals>, editing: bool) {
        if !self.detection {
            set(&mut self.events_strip, format_args!("off"));
        } else if editing {
            set(&mut self.events_strip, format_args!("✏ editing"));
        } else if rules == 0 {
            set(&mut self.events_strip, format_args!("no rules"));
        } else {
            let all = totals.map_or(&[][..], |t| t.rules.as_slice());
            let counted: u64 = all.iter().map(|r| r.a_to_b + r.b_to_a + r.visits).sum();
            set(&mut self.events_strip, format_args!("{counted} counted"));
        }
    }

    /// The strip's Latency row: the two numbers that matter, video latency (App total)
    /// and boxes latency (Frame → boxes), both p50 and rounded to whole ms.
    fn refresh_latency_strip(&mut self, s: &TelemetrySnapshot) {
        let out = &mut self.latency_strip;
        match (s.total, s.detect_total) {
            (Some(video), Some(boxes)) => {
                set(out, format_args!("{:.0} · {:.0} ms", video.p50, boxes.p50));
            }
            (Some(video), None) => set(out, format_args!("{:.0} ms", video.p50)),
            (None, _) => set(out, format_args!("—")),
        }
    }

    /// Whether a detector runs (false with `--no-detect`).
    pub fn detection(&self) -> bool {
        self.detection
    }

    /// The Detect page's rows and the strip's Detect value; `—` while detection is off
    /// or has no result yet (the strip says `off` for the former).
    fn refresh_detector(&mut self, s: &TelemetrySnapshot) {
        let has_result = s.detect_total.is_some();
        if self.detection && has_result {
            set(
                &mut self.detector_fps,
                format_args!("{:.1}", s.detector_fps),
            );
            set(
                &mut self.detect_strip,
                format_args!("{:.1} fps", s.detector_fps),
            );
        } else {
            set(&mut self.detector_fps, format_args!("—"));
            let strip = if self.detection { "—" } else { "off" };
            set(&mut self.detect_strip, format_args!("{strip}"));
        }
        if !self.detection {
            set(&mut self.det_skipped, format_args!("—"));
        } else if s.det_errors > 0 {
            set(
                &mut self.det_skipped,
                format_args!("{} (+{} errors)", s.det_skipped, s.det_errors),
            );
        } else {
            set(&mut self.det_skipped, format_args!("{}", s.det_skipped));
        }
        if self.detection {
            set(&mut self.det_paced, format_args!("{}", s.det_paced));
        } else {
            set(&mut self.det_paced, format_args!("—"));
        }
        match s.boxes_behind {
            Some(n) => set(&mut self.boxes_behind, format_args!("{n}")),
            None => set(&mut self.boxes_behind, format_args!("—")),
        }
        self.refresh_tracking(s, self.detection && has_result);
    }

    /// The tracking and pacing rows; `—` until there is a result to read them from.
    fn refresh_tracking(&mut self, s: &TelemetrySnapshot, has_result: bool) {
        match s.detect_interval.filter(|_| has_result) {
            Some(d) if d.is_zero() => set(&mut self.detect_interval, format_args!("every frame")),
            Some(d) => set(
                &mut self.detect_interval,
                format_args!("{} ms", d.as_millis()),
            ),
            None => set(&mut self.detect_interval, format_args!("—")),
        }
        if !has_result {
            set(&mut self.tracks, format_args!("—"));
            set(&mut self.track_ids, format_args!("—"));
            let strip = if self.detection { "—" } else { "off" };
            set(&mut self.track_strip, format_args!("{strip}"));
            return;
        }
        if s.tracks_lost > 0 {
            set(
                &mut self.tracks,
                format_args!("{} (+{} lost)", s.tracks, s.tracks_lost),
            );
            set(
                &mut self.track_strip,
                format_args!("{} · {} lost", s.tracks, s.tracks_lost),
            );
        } else {
            set(&mut self.tracks, format_args!("{}", s.tracks));
            set(&mut self.track_strip, format_args!("{}", s.tracks));
        }
        set(
            &mut self.track_ids,
            format_args!("{} · {} recovered", s.track_ids, s.recovered),
        );
    }
}

fn set(out: &mut String, args: std::fmt::Arguments<'_>) {
    out.clear();
    let _ = out.write_fmt(args);
}

/// One right-aligned ms value with 1 decimal, or `—` for no data.
fn cell(out: &mut String, v: Option<f32>) {
    match v {
        Some(v) => set(out, format_args!("{v:>CELL$.1}")),
        None => set(out, format_args!("{:>CELL$}", "—")),
    }
}

/// Like [`cell`], but with 2 decimals. Tracking and events take hundredths of a
/// millisecond, so one decimal would always read `0.0`.
fn fine_cell(out: &mut String, v: Option<f32>) {
    match v {
        Some(v) => set(out, format_args!("{v:>CELL$.2}")),
        None => set(out, format_args!("{:>CELL$}", "—")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::stats::Summary;

    fn info() -> SourceInfo {
        SourceInfo {
            name: "cam".into(),
            format: "YUYV".into(),
            width: 1920,
            height: 1080,
            fps: 30,
        }
    }

    #[test]
    fn cells_are_fixed_width() {
        let mut s = String::new();
        cell(&mut s, Some(2.345));
        assert_eq!(s, "    2.3");
        cell(&mut s, None);
        assert_eq!(s.chars().count(), CELL);
        assert!(s.ends_with('—'));
        fine_cell(&mut s, Some(0.0234));
        assert_eq!(s, "   0.02");
    }

    #[test]
    fn dash_text_refresh() {
        let mut t = DashText::new(&info(), true);
        assert_eq!(t.source, "1920×1080 · YUYV · 30 fps");
        assert!(t.latency[TOTAL][2].ends_with('—'));
        let sum = Some(Summary {
            last: 12.0,
            p50: 10.0,
            p95: 20.5,
        });
        t.refresh(&TelemetrySnapshot {
            seq: 5,
            dropped: 2,
            bad_buffers: 1,
            total: sum,
            interval: sum,
            ..Default::default()
        });
        assert_eq!(t.frames, "5");
        assert_eq!(t.dropped, "2 (+1 bad buffers)");
        assert_eq!(t.interval, "12.0 / 20.5 ms");
        assert_eq!(t.latency[TOTAL][2], "   20.5");
        assert_eq!(t.latency[SENSOR][0], "    n/a");
    }

    #[test]
    fn dash_text_detector_rows() {
        let mut t = DashText::new(&info(), true);
        // Running, but no result yet.
        assert!(t.latency[INFER][0].ends_with('—'));
        assert_eq!(
            (t.detector_fps.as_str(), t.boxes_behind.as_str()),
            ("—", "—")
        );
        assert_eq!(t.det_skipped, "0");
        assert_eq!(t.det_paced, "0");
        assert_eq!(
            (
                t.detect_interval.as_str(),
                t.tracks.as_str(),
                t.track_ids.as_str()
            ),
            ("—", "—", "—")
        );
        assert!(t.latency[TRACK][0].ends_with('—'));
        assert!(t.latency[EVENTS][0].ends_with('—'));

        let sum = |v: f32| {
            Some(Summary {
                last: v,
                p50: v,
                p95: v,
            })
        };
        t.refresh(&TelemetrySnapshot {
            seq: 9,
            preprocess: sum(3.0),
            inference: sum(21.24),
            postprocess: sum(1.0),
            detect_total: sum(40.0),
            detector_fps: 29.94,
            det_skipped: 4,
            det_errors: 2,
            det_paced: 11,
            boxes_behind: Some(2),
            tracking: sum(0.0234),
            events: sum(0.01),
            events_dropped: 5,
            tracks: 3,
            tracks_lost: 1,
            track_ids: 7,
            recovered: 2,
            detect_interval: Some(std::time::Duration::from_millis(250)),
            ..Default::default()
        });
        assert_eq!(t.latency[PRE][0], "    3.0");
        assert_eq!(t.latency[INFER][1], "   21.2");
        assert_eq!(t.latency[POST][2], "    1.0");
        assert_eq!(t.latency[DETECT_TOTAL][0], "   40.0");
        assert_eq!(t.detector_fps, "29.9");
        assert_eq!(t.det_skipped, "4 (+2 errors)");
        assert_eq!(t.boxes_behind, "2");
        assert_eq!(t.det_paced, "11");
        assert_eq!(t.latency[TRACK][1], "   0.02");
        assert_eq!(t.latency[EVENTS][0], "   0.01");
        assert_eq!(t.detect_interval, "250 ms");
        assert_eq!(t.tracks, "3 (+1 lost)");
        assert_eq!(t.track_ids, "7 · 2 recovered");

        t.refresh(&TelemetrySnapshot {
            detect_total: sum(40.0),
            tracks: 3,
            detect_interval: Some(std::time::Duration::ZERO),
            ..Default::default()
        });
        assert_eq!(t.tracks, "3");
        assert_eq!(t.detect_interval, "every frame");
    }

    fn summary(p50: f32) -> Option<Summary> {
        Some(Summary {
            last: p50,
            p50,
            p95: p50,
        })
    }

    #[test]
    fn dash_text_strip_camera() {
        let mut t = DashText::new(&info(), true);
        assert_eq!(t.camera_strip, "—");
        t.refresh(&TelemetrySnapshot {
            seq: 3,
            camera_fps: 29.94,
            ..Default::default()
        });
        assert_eq!(t.camera_strip, "29.9 fps");
    }

    #[test]
    fn dash_text_strip_detect_and_track() {
        let mut t = DashText::new(&info(), true);
        assert_eq!(
            (t.detect_strip.as_str(), t.track_strip.as_str()),
            ("—", "—")
        );
        let running = TelemetrySnapshot {
            detect_total: summary(16.0),
            detector_fps: 30.04,
            tracks: 3,
            tracks_lost: 1,
            ..Default::default()
        };
        t.refresh(&running);
        assert_eq!(t.detect_strip, "30.0 fps");
        assert_eq!(t.track_strip, "3 · 1 lost");
        t.refresh(&TelemetrySnapshot {
            tracks_lost: 0,
            ..running
        });
        assert_eq!(t.track_strip, "3");

        let off = DashText::new(&info(), false);
        assert_eq!(
            (off.detect_strip.as_str(), off.track_strip.as_str()),
            ("off", "off")
        );
    }

    #[test]
    fn dash_text_strip_events() {
        use crate::events::event::RuleTotals;
        let rule = |a_to_b, b_to_a, visits| RuleTotals {
            rule: 1,
            a_to_b,
            b_to_a,
            visits,
            inside: Vec::new(),
            last_event: None,
        };
        let totals = EventTotals {
            rules: vec![rule(2, 1, 0), rule(0, 0, 2)],
        };
        let mut t = DashText::new(&info(), true);
        assert_eq!(t.events_strip, "no rules");
        t.refresh_events(2, Some(&totals), false);
        assert_eq!(t.events_strip, "5 counted");
        t.refresh_events(2, None, false);
        assert_eq!(t.events_strip, "0 counted");
        t.refresh_events(2, Some(&totals), true);
        assert_eq!(t.events_strip, "✏ editing");
        t.refresh_events(0, None, false);
        assert_eq!(t.events_strip, "no rules");

        let mut off = DashText::new(&info(), false);
        off.refresh_events(2, Some(&totals), true);
        assert_eq!(off.events_strip, "off");
    }

    #[test]
    fn dash_text_strip_latency() {
        let mut t = DashText::new(&info(), true);
        assert_eq!(t.latency_strip, "—");
        let video = TelemetrySnapshot {
            total: summary(80.4),
            ..Default::default()
        };
        t.refresh(&video);
        assert_eq!(t.latency_strip, "80 ms");
        t.refresh(&TelemetrySnapshot {
            detect_total: summary(15.6),
            ..video
        });
        assert_eq!(t.latency_strip, "80 · 16 ms");
    }

    #[test]
    fn dash_text_detection_off() {
        let t = DashText::new(&info(), false);
        assert_eq!(t.detector_fps, "—");
        assert_eq!(t.det_skipped, "—");
        assert_eq!(t.boxes_behind, "—");
        assert_eq!(t.det_paced, "—");
        assert_eq!(t.detect_interval, "—");
        assert_eq!((t.tracks.as_str(), t.track_ids.as_str()), ("—", "—"));
        assert!(t.latency[DETECT_TOTAL][2].ends_with('—'));
        assert!(t.latency[TRACK][2].ends_with('—'));
        assert!(t.latency[EVENTS][2].ends_with('—'));
    }
}
