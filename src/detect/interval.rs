//! Detect interval: decide **when** the detector should run.
//!
//! The camera delivers a frame every ~33 ms, and the detector could run on every one it
//! can get. But the tracker follows the objects, so it can *predict* where each box is
//! between two detections. Every detector run we skip is then saved compute (and
//! battery), at the price of reacting a little later to things the tracker can't
//! predict.
//!
//! There are three modes ([`DetectInterval`]):
//!
//! | Mode | Behaviour |
//! |---|---|
//! | `EveryFrame` | run on every frame (the default) |
//! | `Fixed(d)` | at most one run per `d` (1–500 ms) |
//! | `Auto` | the controller picks the interval, between 0 and [`MAX_INTERVAL`] (500 ms) |
//!
//! # How Auto decides
//!
//! After each detector run, the tracker reports what happened in an [`IntervalSignal`],
//! and [`IntervalController::on_result`] adjusts the interval:
//!
//! | After a detector run the tracker reports… | Auto does |
//! |---|---|
//! | trouble: a track just went lost, a new object appeared, or mean IoU < 0.5 | interval ÷ 2 (and to 0 when under 20 ms) |
//! | calm: no trouble, and mean IoU ≥ 0.7 or nothing to match | interval + 50 ms, up to 500 ms |
//! | otherwise | keep it |
//!
//! "Mean IoU" compares the box the tracker *predicted* with the box the detector then
//! *found*: near 1 means the predictions held up, low means they missed. Growing
//! slowly and cutting fast is the same idea TCP uses for its sending rate: one sign of
//! trouble is enough to look more often, but it takes many calm runs to earn a long
//! wait. The setting is tuned for compute, not for the quickest reaction.
//!
//! # Why a ceiling
//!
//! The tracker can't predict an object it has never seen. With a 500 ms ceiling,
//! someone walking into an empty room gets a box within half a second. The ceiling
//! also stays below the UI's 1 s rule that hides stale results
//! (`ui::hud::STALL_AFTER`), so a paced detector never looks stalled.
//!
//! # Worked example
//!
//! An empty room: every run is calm (nothing to match), so the interval grows
//! 0 → 50 → 100 → … → 500 ms in 10 calm runs. At 30 fps that takes about 2.3 s, since
//! each wait ends on the next frame to arrive. Then someone enters: the tracker reports
//! `born = 1`, so the interval halves 500 → 250 → 125 while the new track is confirmed,
//! and it grows again, 50 ms per run, while they walk steadily and the predictions hold.

use std::fmt;
use std::str::FromStr;
use std::time::{Duration, Instant};

/// Fixed and Auto ceiling. Below `hud::STALL_AFTER` (1 s), so a paced detector's
/// results never look stale, and short enough that a newcomer is boxed within 0.5 s.
pub const MAX_INTERVAL: Duration = Duration::from_millis(500);
/// Auto's growth per calm run. Compute-first: it reaches the ceiling in 10 calm runs.
const GROW: Duration = Duration::from_millis(50);
/// Below this, halving again is pointless (it's under one 30 fps frame), so Auto
/// drops straight to 0 = every frame.
const FLOOR: Duration = Duration::from_millis(20);
/// Mean IoU (predicted vs detected box) at or above which predictions "held up".
const CALM_IOU: f32 = 0.7;
/// Mean IoU below which predictions clearly missed.
const TROUBLE_IOU: f32 = 0.5;
/// Frames arrive every ~33 ms with a few ms of jitter. Without slack, a 100 ms
/// interval would often see 99.8 ms, skip, and run at 133 ms instead.
const JITTER: Duration = Duration::from_millis(10);

/// How often the detector runs, as the user chose it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DetectInterval {
    /// Run on every frame the detector gets.
    EveryFrame,
    /// Run at most once per this duration (1–500 ms).
    Fixed(Duration),
    /// Let [`IntervalController`] pick the interval from the tracker's feedback.
    Auto,
}

impl FromStr for DetectInterval {
    type Err = String;

    /// Accepts `every`, `auto`, `<n>ms` or `<n>` (any case, surrounding spaces ignored).
    /// `0` means every frame; `1..=500` gives `Fixed(n ms)`.
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let text = s.trim().to_lowercase();
        match text.as_str() {
            "every" => return Ok(Self::EveryFrame),
            "auto" => return Ok(Self::Auto),
            _ => {}
        }
        let digits = text.strip_suffix("ms").unwrap_or(&text);
        let max_ms = MAX_INTERVAL.as_millis() as u64;
        match digits.parse::<u64>() {
            Ok(0) => Ok(Self::EveryFrame),
            Ok(ms) if ms <= max_ms => Ok(Self::Fixed(Duration::from_millis(ms))),
            _ => Err(format!(
                "expected every, auto or a number of ms from 1 to {max_ms}, got {s:?}"
            )),
        }
    }
}

impl fmt::Display for DetectInterval {
    /// Writes `every`, `auto` or `100ms`; each one parses back with `FromStr`.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EveryFrame => write!(f, "every"),
            Self::Auto => write!(f, "auto"),
            Self::Fixed(d) => write!(f, "{}ms", d.as_millis()),
        }
    }
}

/// What the controller learns from one tracker update.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct IntervalSignal {
    /// Confirmed tracks (tracked + lost) after the update.
    pub tracks: usize,
    /// Predicted vs detected box, over matched tracks. `None` when nothing was matched.
    pub mean_iou: Option<f32>,
    /// New tentative tracks: objects the tracker hadn't seen before.
    pub born: usize,
    /// Tracks that just went lost.
    pub lost: usize,
}

/// Paces the detector: says whether a frame is due, and (in Auto) adapts the interval.
pub struct IntervalController {
    mode: DetectInterval,
    /// The wait between runs right now.
    current: Duration,
    /// Frame time of the last run, `None` before the first.
    last_run: Option<Instant>,
}

impl IntervalController {
    pub fn new(mode: DetectInterval) -> Self {
        Self {
            mode,
            current: start_interval(mode),
            last_run: None,
        }
    }

    // The worker passes the mode in every frame (`set_mode`) and never reads it back;
    // only the tests do so far.
    #[allow(dead_code)]
    pub fn mode(&self) -> DetectInterval {
        self.mode
    }

    /// Switches mode. Does nothing if `mode` is unchanged, so calling it every frame
    /// with the same setting doesn't reset Auto. Auto restarts at 0 (every frame): the
    /// safest start until the scene proves calm. The last run time is kept.
    pub fn set_mode(&mut self, mode: DetectInterval) {
        if mode == self.mode {
            return;
        }
        self.mode = mode;
        self.current = start_interval(mode);
    }

    /// The wait between runs right now.
    pub fn current(&self) -> Duration {
        self.current
    }

    /// Should a frame received at `frame_at` run the detector?
    pub fn is_due(&self, frame_at: Instant) -> bool {
        let Some(last_run) = self.last_run else {
            return true;
        };
        if self.current.is_zero() {
            return true;
        }
        // Saturating: frame times only move forward, but never panic if one doesn't.
        let waited = frame_at.saturating_duration_since(last_run);
        waited + JITTER >= self.current
    }

    /// Records that the frame received at `frame_at` ran the detector.
    pub fn ran(&mut self, frame_at: Instant) {
        self.last_run = Some(frame_at);
    }

    /// Feeds back what the tracker saw. Auto adjusts the interval; other modes ignore it.
    pub fn on_result(&mut self, s: IntervalSignal) {
        if self.mode != DetectInterval::Auto {
            return;
        }
        if is_trouble(&s) {
            self.current = halved(self.current);
        } else if is_calm(&s) {
            self.current = (self.current + GROW).min(MAX_INTERVAL);
        }
    }
}

/// The interval a mode starts with: `d` for `Fixed(d)`, otherwise 0 (every frame).
fn start_interval(mode: DetectInterval) -> Duration {
    match mode {
        DetectInterval::Fixed(d) => d,
        DetectInterval::EveryFrame | DetectInterval::Auto => Duration::ZERO,
    }
}

/// Half of `d`, or 0 once that falls under `FLOOR`.
fn halved(d: Duration) -> Duration {
    let half = d / 2;
    if half < FLOOR { Duration::ZERO } else { half }
}

/// A track went lost, a new object appeared, or the predictions missed.
fn is_trouble(s: &IntervalSignal) -> bool {
    s.lost > 0 || s.born > 0 || s.mean_iou.is_some_and(|m| m < TROUBLE_IOU)
}

/// No trouble, and the predictions held up well.
fn is_calm(s: &IntervalSignal) -> bool {
    // No matched tracks (an empty scene) counts as calm: nothing to lose by waiting.
    !is_trouble(s) && s.mean_iou.is_none_or(|m| m >= CALM_IOU)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ms(n: u64) -> Duration {
        Duration::from_millis(n)
    }

    fn with_iou(m: f32) -> IntervalSignal {
        IntervalSignal {
            tracks: 1,
            mean_iou: Some(m),
            ..IntervalSignal::default()
        }
    }

    /// An Auto controller grown to `target` by calm runs (a multiple of `GROW`).
    fn auto_at(target: Duration) -> IntervalController {
        let mut c = IntervalController::new(DetectInterval::Auto);
        while c.current() < target {
            c.on_result(with_iou(0.9));
        }
        assert_eq!(c.current(), target);
        c
    }

    #[test]
    fn interval_parses_every_auto_and_ms() {
        let parse = |s: &str| s.parse::<DetectInterval>();
        assert_eq!(parse("every"), Ok(DetectInterval::EveryFrame));
        assert_eq!(parse(" Auto "), Ok(DetectInterval::Auto));
        assert_eq!(parse("100ms"), Ok(DetectInterval::Fixed(ms(100))));
        assert_eq!(parse("100"), Ok(DetectInterval::Fixed(ms(100))));
        assert_eq!(parse("0"), Ok(DetectInterval::EveryFrame));
        for bad in ["501", "2s", "", "-5"] {
            let err = parse(bad).unwrap_err();
            assert!(err.contains("1 to 500"), "{bad:?}: {err}");
        }
    }

    #[test]
    fn interval_display_round_trips() {
        for mode in [
            DetectInterval::EveryFrame,
            DetectInterval::Auto,
            DetectInterval::Fixed(ms(250)),
        ] {
            assert_eq!(mode.to_string().parse::<DetectInterval>(), Ok(mode));
        }
        assert_eq!(DetectInterval::Fixed(ms(250)).to_string(), "250ms");
    }

    #[test]
    fn interval_every_frame_is_always_due() {
        let t0 = Instant::now();
        let mut c = IntervalController::new(DetectInterval::EveryFrame);
        assert!(c.is_due(t0));
        c.ran(t0);
        assert!(c.is_due(t0));
        assert!(c.is_due(t0 + ms(1)));
    }

    #[test]
    fn interval_fixed_waits_with_jitter_slack() {
        let t0 = Instant::now();
        let mut c = IntervalController::new(DetectInterval::Fixed(ms(100)));
        assert!(c.is_due(t0), "the first frame always runs");
        c.ran(t0);
        assert!(!c.is_due(t0 + ms(33)));
        assert!(!c.is_due(t0 + ms(66)));
        assert!(c.is_due(t0 + ms(95)), "inside the jitter slack");
        let t1 = t0 + ms(95);
        c.ran(t1);
        assert!(!c.is_due(t1 + ms(33)));
        assert!(c.is_due(t1 + ms(100)));
    }

    #[test]
    fn interval_auto_grows_when_calm_and_caps() {
        let mut c = IntervalController::new(DetectInterval::Auto);
        assert_eq!(c.current(), Duration::ZERO);
        for _ in 0..20 {
            c.on_result(with_iou(0.9));
            assert!(c.current() <= MAX_INTERVAL);
        }
        assert_eq!(c.current(), MAX_INTERVAL);
    }

    #[test]
    fn interval_auto_empty_scene_is_calm() {
        let mut c = IntervalController::new(DetectInterval::Auto);
        c.on_result(IntervalSignal::default());
        assert_eq!(c.current(), GROW);
    }

    #[test]
    fn interval_auto_halves_on_trouble() {
        let mut c = auto_at(ms(400));
        c.on_result(IntervalSignal {
            lost: 1,
            ..IntervalSignal::default()
        });
        assert_eq!(c.current(), ms(200));
        c.on_result(IntervalSignal {
            born: 1,
            ..IntervalSignal::default()
        });
        assert_eq!(c.current(), ms(100));
        for expected in [50, 25, 0] {
            c.on_result(with_iou(0.3));
            assert_eq!(c.current(), ms(expected));
        }
    }

    #[test]
    fn interval_auto_holds_in_between() {
        let mut c = auto_at(ms(200));
        c.on_result(with_iou(0.6));
        assert_eq!(c.current(), ms(200));
    }

    #[test]
    fn interval_fixed_ignores_signals() {
        let mut c = IntervalController::new(DetectInterval::Fixed(ms(100)));
        c.on_result(IntervalSignal {
            lost: 1,
            born: 1,
            mean_iou: Some(0.1),
            tracks: 1,
        });
        assert_eq!(c.current(), ms(100));
    }

    #[test]
    fn interval_set_mode_resets() {
        let mut c = auto_at(ms(300));
        c.set_mode(DetectInterval::Auto);
        assert_eq!(c.current(), ms(300), "an unchanged mode keeps its interval");
        c.set_mode(DetectInterval::Fixed(ms(50)));
        assert_eq!(c.mode(), DetectInterval::Fixed(ms(50)));
        assert_eq!(c.current(), ms(50));
        c.set_mode(DetectInterval::Auto);
        assert_eq!(c.current(), Duration::ZERO, "Auto restarts at every frame");
    }

    #[test]
    fn interval_max_is_below_stall() {
        assert!(MAX_INTERVAL < crate::ui::hud::STALL_AFTER);
    }
}
