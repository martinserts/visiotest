//! Per-stage timings of shown frames, rolled up into a plain `Copy` snapshot for the UI.
//!
//! Every measured stage pushes into its own [`RollingStats`] here: the video path
//! from [`Telemetry::on_frame`], the detector's stages, tracking and the event engine
//! from [`Telemetry::on_detections`]. A new stage adds its timing the same way, plus a
//! row in the panel's Latency table.

use std::time::{Duration, Instant};

use crate::detect::worker::Detections;
use crate::stats::{FpsCounter, RollingStats, Summary, ms};

/// Timing of the last frame that reached the GPU (copied out; the `Buffer` is released).
#[derive(Clone, Copy, Debug)]
pub struct Shown {
    pub seq: u64,
    pub interval: Option<Duration>,
    pub camera_fps: f64,
    pub sensor_age: Option<Duration>,
    pub upload: Duration,
    pub received: Instant,
}

/// Counters that live outside `Telemetry` (in the slots and capture/detector threads),
/// read once per snapshot.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Counters {
    /// UI slot overwrites.
    pub dropped: u64,
    /// Buffers the capture callback rejected.
    pub bad_buffers: u64,
    /// Detector slot overwrites: frames the detector never saw.
    pub det_skipped: u64,
    /// Frames the detector failed on.
    pub det_errors: u64,
    /// Frames the detector took but skipped because the detect interval said wait.
    pub det_paced: u64,
    /// Events the sink dropped (full queue or a failed write).
    pub events_dropped: u64,
}

/// What the dashboard shows, refreshed at 4 Hz. Plain data, no egui types.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct TelemetrySnapshot {
    /// Camera sequence number of the last shown frame (0 = none yet).
    pub seq: u64,
    /// Frames overwritten in the slot before the UI took them.
    pub dropped: u64,
    /// Camera buffers rejected by the capture callback.
    pub bad_buffers: u64,
    pub camera_fps: f64,
    pub display_fps: f64,
    pub interval: Option<Summary>,
    /// Sensor timestamp → capture callback.
    pub sensor: Option<Summary>,
    /// Whether the last frame carried a sensor timestamp.
    pub has_sensor: bool,
    /// Capture callback → the `ui()` call that took the frame from the slot.
    pub wait: Option<Summary>,
    /// CPU time of `write_texture` + conversion-pass submit.
    pub upload: Option<Summary>,
    /// Sum of the measured stages, per frame.
    pub total: Option<Summary>,
    /// The last total excluded `sensor` (no timestamp).
    pub total_partial: bool,

    /// Detector: YUYV → input tensor.
    pub preprocess: Option<Summary>,
    /// Detector: the `session.run` call.
    pub inference: Option<Summary>,
    /// Detector: decode + NMS + un-letterbox.
    pub postprocess: Option<Summary>,
    /// Frame arrival (`Frame::received`) → its boxes and tracks are ready.
    pub detect_total: Option<Summary>,
    /// Detector: `Tracker::update` + `views`.
    pub tracking: Option<Summary>,
    /// Detector: rule engine (set rules + update + totals).
    pub events: Option<Summary>,
    /// Rate at which detection results reach the UI.
    pub detector_fps: f64,
    /// Camera frames the detector never saw (its slot was overwritten while it was busy).
    pub det_skipped: u64,
    /// Frames the detector failed on.
    pub det_errors: u64,
    /// Frames the detector skipped on purpose, because the detect interval said wait.
    pub det_paced: u64,
    /// How many camera frames the newest boxes trail the shown video by.
    pub boxes_behind: Option<u64>,
    /// Boxes in the newest result (all scores ≥ `LOW_CONF`).
    pub boxes: usize,
    /// Confirmed tracks matched in the newest result.
    pub tracks: usize,
    /// Confirmed tracks currently lost (hidden or missed).
    pub tracks_lost: usize,
    /// Track IDs handed out so far.
    pub track_ids: u32,
    /// Lost tracks found again, since startup: occlusions the tracker survived.
    pub recovered: u64,
    /// The wait between detector runs (0 = every frame); `None` before the first result.
    pub detect_interval: Option<Duration>,
    /// Events lost on the way to stdout (full queue or a failed write).
    pub events_dropped: u64,
}

pub struct Telemetry {
    sensor: RollingStats,
    wait: RollingStats,
    upload: RollingStats,
    total: RollingStats,
    interval: RollingStats,
    display: FpsCounter,
    display_fps: f64,
    seq: u64,
    camera_fps: f64,
    has_sensor: bool,
    preprocess: RollingStats,
    inference: RollingStats,
    postprocess: RollingStats,
    detect_total: RollingStats,
    detector: FpsCounter,
    detector_fps: f64,
    /// Camera seq of the newest detection result (0 = none yet).
    det_seq: u64,
    boxes: usize,
    tracking: RollingStats,
    events: RollingStats,
    tracked: usize,
    lost: usize,
    ids: u32,
    /// Running total of `TrackStep::recovered`.
    recovered: u64,
    /// The newest result's detect interval (`interval` is the camera's frame interval).
    detect_interval: Option<Duration>,
}

impl Telemetry {
    pub fn new() -> Self {
        Self {
            sensor: RollingStats::new(),
            wait: RollingStats::new(),
            upload: RollingStats::new(),
            total: RollingStats::new(),
            interval: RollingStats::new(),
            display: FpsCounter::new(),
            display_fps: 0.0,
            seq: 0,
            camera_fps: 0.0,
            has_sensor: false,
            preprocess: RollingStats::new(),
            inference: RollingStats::new(),
            postprocess: RollingStats::new(),
            detect_total: RollingStats::new(),
            detector: FpsCounter::new(),
            detector_fps: 0.0,
            det_seq: 0,
            boxes: 0,
            tracking: RollingStats::new(),
            events: RollingStats::new(),
            tracked: 0,
            lost: 0,
            ids: 0,
            recovered: 0,
            detect_interval: None,
        }
    }

    /// Record a frame that was just uploaded. `taken_at` is when `ui()` took it from the slot.
    pub fn on_frame(&mut self, f: &Shown, taken_at: Instant) {
        let wait = taken_at.saturating_duration_since(f.received);
        let mut total = wait + f.upload;
        if let Some(age) = f.sensor_age {
            self.sensor.push(ms(age) as f32);
            total += age;
        }
        self.wait.push(ms(wait) as f32);
        self.upload.push(ms(f.upload) as f32);
        self.total.push(ms(total) as f32);
        if let Some(dt) = f.interval {
            self.interval.push(ms(dt) as f32);
        }
        self.seq = f.seq;
        self.camera_fps = f.camera_fps;
        self.has_sensor = f.sensor_age.is_some();
    }

    /// Tick the display-rate counter; call once per `ui()`.
    pub fn on_ui(&mut self, now: Instant) {
        self.display_fps = self.display.tick(now);
    }

    /// Record a detection result the UI just took from the detector's slot.
    pub fn on_detections(&mut self, d: &Detections, now: Instant) {
        self.preprocess.push(ms(d.preprocess) as f32);
        self.inference.push(ms(d.inference) as f32);
        self.postprocess.push(ms(d.postprocess) as f32);
        // From the moment the frame reached user space, so it includes the time the
        // frame waited in the detector's slot, not just the three stages.
        let total = d.done.saturating_duration_since(d.received);
        self.detect_total.push(ms(total) as f32);
        // The UI polls far faster than the detector runs, so the rate results arrive
        // here is the detector's own rate.
        self.detector_fps = self.detector.tick(now);
        self.det_seq = d.seq;
        self.boxes = d.boxes.len();
        self.tracking.push(ms(d.tracking) as f32);
        self.events.push(ms(d.event_time) as f32);
        self.tracked = d.step.tracked;
        self.lost = d.step.lost;
        self.ids = d.ids;
        self.recovered += d.step.recovered as u64;
        self.detect_interval = Some(d.interval);
    }

    pub fn snapshot(&mut self, c: Counters) -> TelemetrySnapshot {
        // Both seqs count camera frames, so their difference is in frames. Boxes can
        // be *ahead* for a moment (the detector finished a frame the UI hasn't shown
        // yet), which reads as 0.
        let boxes_behind =
            (self.seq > 0 && self.det_seq > 0).then(|| self.seq.saturating_sub(self.det_seq));
        TelemetrySnapshot {
            seq: self.seq,
            dropped: c.dropped,
            bad_buffers: c.bad_buffers,
            camera_fps: self.camera_fps,
            display_fps: self.display_fps,
            interval: self.interval.summary(),
            sensor: self.sensor.summary(),
            has_sensor: self.has_sensor,
            wait: self.wait.summary(),
            upload: self.upload.summary(),
            total: self.total.summary(),
            total_partial: self.seq > 0 && !self.has_sensor,
            preprocess: self.preprocess.summary(),
            inference: self.inference.summary(),
            postprocess: self.postprocess.summary(),
            detect_total: self.detect_total.summary(),
            tracking: self.tracking.summary(),
            events: self.events.summary(),
            detector_fps: self.detector_fps,
            det_skipped: c.det_skipped,
            det_errors: c.det_errors,
            det_paced: c.det_paced,
            boxes_behind,
            boxes: self.boxes,
            tracks: self.tracked,
            tracks_lost: self.lost,
            track_ids: self.ids,
            recovered: self.recovered,
            detect_interval: self.detect_interval,
            events_dropped: c.events_dropped,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::events::event::EventTotals;
    use crate::track::tracker::TrackStep;

    fn shown(seq: u64, received: Instant, sensor_ms: Option<u64>, upload_ms: u64) -> Shown {
        Shown {
            seq,
            interval: (seq > 1).then(|| Duration::from_millis(33)),
            camera_fps: 30.0,
            sensor_age: sensor_ms.map(Duration::from_millis),
            upload: Duration::from_millis(upload_ms),
            received,
        }
    }

    fn close(a: f32, b: f32) -> bool {
        (a - b).abs() < 1e-3
    }

    #[test]
    fn telemetry_rows_from_two_frames() {
        let mut t = Telemetry::new();
        let t0 = Instant::now();
        t.on_frame(&shown(1, t0, Some(10), 1), t0 + Duration::from_millis(4));
        let t1 = t0 + Duration::from_millis(33);
        t.on_frame(&shown(2, t1, Some(20), 2), t1 + Duration::from_millis(8));
        let s = t.snapshot(Counters::default());
        assert_eq!(s.seq, 2);
        assert!(s.has_sensor && !s.total_partial);
        assert!(close(s.sensor.unwrap().last, 20.0));
        assert!(close(s.sensor.unwrap().p50, 10.0));
        assert!(close(s.wait.unwrap().last, 8.0));
        assert!(close(s.upload.unwrap().last, 2.0));
        // Frame 1 had no interval, so only one sample.
        assert!(close(s.interval.unwrap().p50, 33.0));
    }

    #[test]
    fn telemetry_total_is_sum_of_parts() {
        let mut t = Telemetry::new();
        let t0 = Instant::now();
        t.on_frame(&shown(1, t0, Some(12), 3), t0 + Duration::from_millis(5));
        let total = t.snapshot(Counters::default()).total.unwrap();
        assert!(close(total.last, 12.0 + 5.0 + 3.0), "{total:?}");
    }

    #[test]
    fn telemetry_missing_sensor_flags_partial_total() {
        let mut t = Telemetry::new();
        let t0 = Instant::now();
        t.on_frame(&shown(1, t0, None, 3), t0 + Duration::from_millis(5));
        let s = t.snapshot(Counters::default());
        assert!(s.total_partial && !s.has_sensor);
        assert_eq!(s.sensor, None);
        assert!(close(s.total.unwrap().last, 8.0));
    }

    #[test]
    fn telemetry_snapshot_carries_counters() {
        let mut t = Telemetry::new();
        let s = t.snapshot(Counters {
            dropped: 7,
            bad_buffers: 2,
            det_skipped: 5,
            det_errors: 1,
            det_paced: 0,
            events_dropped: 3,
        });
        assert_eq!((s.dropped, s.bad_buffers, s.seq), (7, 2, 0));
        assert_eq!((s.det_skipped, s.det_errors), (5, 1));
        assert_eq!(s.events_dropped, 3);
        assert!(!s.total_partial && s.total.is_none());
        assert_eq!((s.boxes_behind, s.inference), (None, None));
    }

    fn detections(seq: u64, received: Instant, stages_ms: [u64; 3], boxes: usize) -> Detections {
        let [pre, infer, post] = stages_ms.map(Duration::from_millis);
        let bbox = crate::detect::BBox::new(0.0, 0.0, 1.0, 1.0);
        let one = crate::detect::Detection {
            bbox,
            class: 0,
            score: 0.9,
        };
        Detections {
            seq,
            received,
            preprocess: pre,
            inference: infer,
            postprocess: post,
            // 5 ms waiting in the detector slot, then the three stages.
            done: received + Duration::from_millis(5) + pre + infer + post,
            boxes: vec![one; boxes],
            tracking: Duration::ZERO,
            tracks: Vec::new(),
            step: TrackStep::default(),
            ids: 0,
            interval: Duration::ZERO,
            totals: EventTotals::default(),
            event_time: Duration::ZERO,
        }
    }

    #[test]
    fn telemetry_tracking_rows() {
        let mut t = Telemetry::new();
        let t0 = Instant::now();
        assert_eq!(t.snapshot(Counters::default()).detect_interval, None);
        let step = TrackStep {
            tracked: 3,
            lost: 1,
            recovered: 1,
            ..TrackStep::default()
        };
        let first = Detections {
            tracking: Duration::from_micros(20),
            step,
            ids: 5,
            interval: Duration::from_millis(250),
            ..detections(1, t0, [1, 1, 1], 0)
        };
        t.on_detections(&first, t0);
        let second = Detections {
            tracking: Duration::from_micros(30),
            ..first.clone()
        };
        t.on_detections(&second, t0 + Duration::from_millis(250));
        let s = t.snapshot(Counters {
            det_paced: 6,
            ..Counters::default()
        });
        assert!(close(s.tracking.unwrap().last, 0.03));
        assert_eq!((s.tracks, s.tracks_lost, s.track_ids), (3, 1, 5));
        assert_eq!(s.recovered, 2); // one per result, added up
        assert_eq!(s.det_paced, 6);
        assert_eq!(s.detect_interval, Some(Duration::from_millis(250)));
    }

    #[test]
    fn telemetry_events_row() {
        let mut t = Telemetry::new();
        let t0 = Instant::now();
        assert_eq!(t.snapshot(Counters::default()).events, None);
        let d = Detections {
            event_time: Duration::from_micros(10),
            ..detections(1, t0, [1, 1, 1], 0)
        };
        t.on_detections(&d, t0);
        let s = t.snapshot(Counters::default());
        assert!(close(s.events.unwrap().last, 0.01));
    }

    #[test]
    fn telemetry_detection_rows() {
        let mut t = Telemetry::new();
        let t0 = Instant::now();
        t.on_frame(&shown(10, t0, Some(10), 1), t0 + Duration::from_millis(4));
        t.on_detections(&detections(7, t0, [3, 20, 2], 4), t0);
        let s = t.snapshot(Counters::default());
        assert!(close(s.preprocess.unwrap().last, 3.0));
        assert!(close(s.inference.unwrap().last, 20.0));
        assert!(close(s.postprocess.unwrap().last, 2.0));
        assert!(close(s.detect_total.unwrap().last, 5.0 + 3.0 + 20.0 + 2.0));
        assert_eq!(s.boxes_behind, Some(3));
        assert_eq!(s.boxes, 4);
    }

    #[test]
    fn telemetry_boxes_ahead_of_video_read_as_zero() {
        let mut t = Telemetry::new();
        let t0 = Instant::now();
        t.on_frame(&shown(5, t0, None, 1), t0);
        t.on_detections(&detections(6, t0, [1, 1, 1], 0), t0);
        assert_eq!(t.snapshot(Counters::default()).boxes_behind, Some(0));
    }

    #[test]
    fn telemetry_detector_fps() {
        let mut t = Telemetry::new();
        let t0 = Instant::now();
        // A result every 50 ms is 20 fps. The counter's first 1 s window closes on the
        // 21st tick (it counts both ends), so allow a little slack.
        for i in 0..=20 {
            let now = t0 + Duration::from_millis(50 * i);
            t.on_detections(&detections(i + 1, now, [1, 1, 1], 0), now);
        }
        let fps = t.snapshot(Counters::default()).detector_fps;
        assert!((fps - 20.0).abs() < 1.5, "{fps}");
    }
}
