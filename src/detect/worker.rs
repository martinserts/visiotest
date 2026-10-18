//! The detector worker: one thread that turns camera frames into boxes, tracks and
//! events.
//!
//! ```text
//! capture ─► Latest<Frame> ─► [detector thread: pace? → preprocess → model
//!                                   → postprocess → tracker → events]
//!                                   ─► Latest<Detections> ─► UI
//!                                   └► EventSender ─► "events" thread ─► stdout
//! ```
//!
//! Frames and results go through latest-only slots, and the detector skips frames for
//! two reasons:
//!
//! - **Busy** (forced): while inference runs, the frames that arrive are overwritten in
//!   its slot, so the detector always works on the newest frame and never builds up a
//!   backlog. These are the slot's `dropped()` count.
//! - **Paced** (chosen): the [`IntervalController`] says the next run isn't due yet. The
//!   worker takes the frame and drops it before doing any work, and counts it in
//!   [`DetectorCounters::paced`]. Between runs the tracker's predictions fill the gap.
//!
//! The UI never waits for the detector: it only `take()`s whatever result is ready.
//!
//! The event engine runs last, on the tracks of the same run, and each event also goes
//! to the [`EventSender`] (when `--events stdout`): a bounded queue that never blocks
//! this thread, because an overwritten event would be a lost count (see [`crate::emit`]).

use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use anyhow::{Context, Result};

use super::interval::{DetectInterval, IntervalController};
use super::model::Model;
use super::postprocess::Postprocessor;
use super::preprocess::{Preprocessor, TENSOR_LEN};
use super::{Detection, MAX_DETECTIONS};
use crate::capture::{Frame, FrameSlot, YuyvLayout};
use crate::color::ColorMatrix;
use crate::emit::{EventSender, EventSink};
use crate::events::engine::{EngineConfig, EventEngine};
use crate::events::event::EventTotals;
use crate::events::rules::Rules;
use crate::slot::Latest;
use crate::track::tracker::{LOW_CONF, TrackStep, TrackView, Tracker, TrackerConfig};

/// How long the worker waits for a frame before checking its stop flag again.
const POLL: Duration = Duration::from_millis(100);
/// Default confidence threshold: Ultralytics predicts with 0.25; a bit higher keeps
/// the live view free of flickering low-confidence boxes.
const DEFAULT_CONF: f32 = 0.35;

/// The boxes, tracks and events for one camera frame, with the timings of each stage
/// that made them.
#[derive(Clone, Debug)]
pub struct Detections {
    /// Camera seq of the frame these boxes belong to.
    pub seq: u64,
    /// That frame's `Frame::received`.
    pub received: Instant,
    pub preprocess: Duration,
    /// The `session.run` call only.
    pub inference: Duration,
    pub postprocess: Duration,
    /// When the result (boxes, tracks and events) was ready.
    pub done: Instant,
    /// After NMS: all classes, score ≥ `LOW_CONF` (the UI filters by the confidence
    /// slider), best first.
    pub boxes: Vec<Detection>,
    /// `Tracker::update` + `views`.
    pub tracking: Duration,
    /// Every confirmed track (Tracked and Lost) after this frame, by id.
    pub tracks: Vec<TrackView>,
    /// What this tracker update did.
    pub step: TrackStep,
    /// Track IDs handed out so far (`Tracker::ids_issued`).
    pub ids: u32,
    /// The wait between detector runs after this result (0 = every frame).
    pub interval: Duration,
    /// Every rule's running totals after this run. Cumulative, so the UI's counts stay
    /// exact even when a result is overwritten in the slot before the UI takes it.
    pub totals: EventTotals,
    /// The rule engine: `set_rules` (when the rules or the frame size changed),
    /// `EventEngine::update` and `totals`.
    pub event_time: Duration,
}

/// Counters the worker bumps and the UI reads.
#[derive(Default)]
pub struct DetectorCounters {
    /// Frames the detector failed on.
    pub errors: AtomicU64,
    /// Frames it took from its slot but didn't run on, because the interval said wait.
    pub paced: AtomicU64,
}

/// Settings the UI can change while the detector runs.
///
/// The confidence threshold is an `f32` stored as its bit pattern in an `AtomicU32`:
/// std has no `AtomicF32`, and a lock would be overkill for one number.
///
/// The detect interval is behind a `Mutex` instead: an enum holding a `Duration`
/// doesn't fit in one atomic word. It is read once per frame and the lock is only held
/// for the copy, never while working, so neither thread ever waits on it for long.
///
/// The rules are behind a `Mutex` for the same reason (they hold `Vec`s), together with
/// a **version** that every `set_rules` bumps. The detector compares versions first and
/// clones only when the version moved, so a frame normally costs one lock and one
/// integer compare, not a copy of the rules.
///
/// "Reset counts" is a **generation** counter: the UI bumps it, and the detector resets
/// its totals whenever the number differs from the last one it acted on. A counter
/// (rather than a flag the detector clears) needs no write back from the detector, and
/// two quick presses still reset at least once.
pub struct DetectorSettings {
    conf_bits: AtomicU32,
    interval: Mutex<DetectInterval>,
    /// (version, rules); version 0 means "never set".
    rules: Mutex<(u64, Rules)>,
    reset: AtomicU64,
}

impl Default for DetectorSettings {
    fn default() -> Self {
        Self::new(DetectInterval::EveryFrame)
    }
}

impl DetectorSettings {
    /// The default confidence threshold and the given detect interval.
    pub fn new(interval: DetectInterval) -> Self {
        Self {
            conf_bits: AtomicU32::new(DEFAULT_CONF.to_bits()),
            interval: Mutex::new(interval),
            rules: Mutex::new((0, Rules::default())),
            reset: AtomicU64::new(0),
        }
    }

    pub fn conf(&self) -> f32 {
        f32::from_bits(self.conf_bits.load(Ordering::Relaxed))
    }

    pub fn set_conf(&self, conf: f32) {
        self.conf_bits.store(conf.to_bits(), Ordering::Relaxed);
    }

    /// When the detector should run.
    pub fn interval(&self) -> DetectInterval {
        // Poison-tolerant: a panic elsewhere can't leave a plain enum half-written.
        *self.interval.lock().unwrap_or_else(|e| e.into_inner())
    }

    // The Track page's Detect control is the only caller; the CLI sets the start
    // value through `new`.
    pub fn set_interval(&self, interval: DetectInterval) {
        *self.interval.lock().unwrap_or_else(|e| e.into_inner()) = interval;
    }

    /// Hands the detector a new rule set: a copy, under a new version. The detector
    /// adopts it on its next run.
    pub fn set_rules(&self, rules: &Rules) {
        let mut guard = self.rules.lock().unwrap_or_else(|e| e.into_inner());
        guard.0 += 1;
        guard.1 = rules.clone();
    }

    /// The rules and their version, if the version isn't `seen` (the one the caller
    /// already has). `None` costs no clone.
    pub fn rules_if_newer(&self, seen: u64) -> Option<(u64, Rules)> {
        let guard = self.rules.lock().unwrap_or_else(|e| e.into_inner());
        let (version, rules) = &*guard;
        (*version != seen).then(|| (*version, rules.clone()))
    }

    /// Asks the detector to zero every rule's counts on its next run.
    pub fn request_reset(&self) {
        self.reset.fetch_add(1, Ordering::Relaxed);
    }

    /// How many resets were requested so far.
    pub fn reset_generation(&self) -> u64 {
        self.reset.load(Ordering::Relaxed)
    }
}

/// Everything the worker thread owns. Allocated once, reused for every frame.
struct WorkerState {
    model: Model,
    matrix: ColorMatrix,
    /// Built from the first frame's layout, rebuilt if the layout ever changes.
    pre: Option<Preprocessor>,
    post: Postprocessor,
    /// The `1×3×640×640` input tensor.
    input: Vec<f32>,
    /// Decides which frames to run on.
    pacing: IntervalController,
    /// Links each run's boxes into tracks; it keeps its buffers between runs.
    tracker: Tracker,
    /// Turns the tracks into events.
    events: EventsState,
}

/// The event stage's part of the worker: the engine and what it has adopted so far.
/// Its own struct so [`sync_rules`] can be tested without a `Model`.
struct EventsState {
    engine: EventEngine,
    /// The detector's copy of the rules (the UI has its own).
    rules: Rules,
    /// The `DetectorSettings` rules version `rules` came from (0 = none yet).
    rules_seen: u64,
    /// The reset generation already acted on.
    reset_seen: u64,
    /// Where each event goes besides the result; `None` with `--events off`.
    out: Option<EventSender>,
}

impl EventsState {
    fn new(out: Option<EventSender>) -> Self {
        Self {
            engine: EventEngine::new(EngineConfig::default()),
            rules: Rules::default(),
            rules_seen: 0,
            reset_seen: 0,
            out,
        }
    }
}

/// Owns the detector thread and the event sink it feeds. Dropping it stops the thread
/// and waits for it, then drops the sink.
pub struct Detector {
    stop: Arc<AtomicBool>,
    thread: Option<JoinHandle<()>>,
    /// Dropped after `drop` has joined the thread (fields drop after the `Drop` body),
    /// which matters: the sink's drop waits for every `EventSender` to be gone, and the
    /// thread holds one.
    _events: Option<EventSink>,
}

impl Detector {
    /// Starts the `"detector"` thread. It takes frames from `frames`, publishes results
    /// to `out`, reads the threshold, interval, rules and reset requests from
    /// `settings`, counts failed and paced frames in `counters`, and sends every event
    /// to `events` (if given), which the `Detector` owns from then on.
    pub fn spawn(
        model: Model,
        matrix: ColorMatrix,
        frames: FrameSlot,
        out: Latest<Detections>,
        settings: Arc<DetectorSettings>,
        counters: Arc<DetectorCounters>,
        events: Option<EventSink>,
    ) -> Result<Detector> {
        let stop = Arc::new(AtomicBool::new(false));
        let state = WorkerState {
            model,
            matrix,
            pre: None,
            post: Postprocessor::new(),
            input: vec![0.0; TENSOR_LEN],
            pacing: IntervalController::new(settings.interval()),
            tracker: Tracker::new(TrackerConfig::default()),
            events: EventsState::new(events.as_ref().map(EventSink::sender)),
        };
        let thread_stop = stop.clone();
        let thread = thread::Builder::new()
            .name("detector".into())
            .spawn(move || run(state, &frames, &out, &settings, &counters, &thread_stop))
            .context("starting the detector thread")?;
        Ok(Detector {
            stop,
            thread: Some(thread),
            _events: events,
        })
    }
}

impl Drop for Detector {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        // The thread only blocks in `wait_take` (at most `POLL`) and in one model run,
        // so this join returns quickly.
        if let Some(t) = self.thread.take() {
            let _ = t.join();
        }
    }
}

/// The thread's loop: wait for a frame, pace, detect + track + events, publish; repeat
/// until stopped.
fn run(
    mut state: WorkerState,
    frames: &FrameSlot,
    out: &Latest<Detections>,
    settings: &DetectorSettings,
    counters: &DetectorCounters,
    stop: &AtomicBool,
) {
    while !stop.load(Ordering::Relaxed) {
        let Some(frame) = frames.wait_take(POLL) else {
            continue;
        };
        // Pace before any work, so a paced frame costs nothing but the slot `take`. It
        // was a refcount clone of the UI's frame, so dropping it frees no pixels.
        if !should_run(&mut state.pacing, settings.interval(), frame.received) {
            counters.paced.fetch_add(1, Ordering::Relaxed);
            continue;
        }
        match detect_one(&mut state, &frame, settings) {
            Ok(d) => out.put(d),
            Err(e) => {
                // Print only the first error: a broken model would fail on every frame
                // and flood the terminal. The status bar shows that errors continue.
                if counters.errors.fetch_add(1, Ordering::Relaxed) == 0 {
                    eprintln!("detector: {e:#}");
                }
            }
        }
    }
}

/// Whether to run on a frame received at `frame_at`. Picks up a changed mode first,
/// and records the run when it says yes.
fn should_run(pacing: &mut IntervalController, mode: DetectInterval, frame_at: Instant) -> bool {
    pacing.set_mode(mode);
    let due = pacing.is_due(frame_at);
    if due {
        pacing.ran(frame_at);
    }
    due
}

/// One frame through the whole pipeline: detect, track, then turn tracks into events.
///
/// The confidence slider (`settings.conf()`) is the tracker's *high* threshold; the
/// boxes themselves are kept down to [`LOW_CONF`].
fn detect_one(
    state: &mut WorkerState,
    frame: &Frame,
    settings: &DetectorSettings,
) -> Result<Detections> {
    let conf = settings.conf();
    let pre = preprocessor_for(&mut state.pre, frame.layout, state.matrix);

    let t0 = Instant::now();
    pre.run(frame.buffer.buffer(), &mut state.input); // YUYV → letterboxed RGB tensor
    let t1 = Instant::now();
    let output = state.model.run(&state.input)?; // the model itself, nothing else
    let t2 = Instant::now();
    // A per-result allocation: the UI keeps this `Vec` (≤ MAX_DETECTIONS).
    let mut boxes = Vec::with_capacity(MAX_DETECTIONS);
    // ByteTrack's second round needs the 0.10–conf boxes too; the UI hides them.
    let floor = LOW_CONF.min(conf);
    state
        .post
        .run(output.raw()?, floor, &pre.letterbox(), &mut boxes); // decode + NMS
    let t3 = Instant::now();
    // The frame's time, not `Instant::now()`: the Kalman filter's `dt` must be the time
    // between the *frames*, not between the detector runs (which add inference jitter).
    let step = state.tracker.update(&boxes, conf, frame.received);
    // Another per-result allocation: one entry per live track.
    let mut tracks = Vec::new();
    state.tracker.views(&mut tracks);
    state.pacing.on_result(step.interval_signal());
    let t4 = Instant::now();

    sync_rules(&mut state.events, settings, frame.layout); // new rules / a reset / a new frame size
    // Allocates only when something happened (`Vec::new()` doesn't allocate). The
    // views' boxes are at `frame.received`, the same time base as the tracker's.
    let mut events = Vec::new();
    state
        .events
        .engine
        .update(&tracks, frame.received, frame.seq, &mut events);
    if let Some(out) = &state.events.out {
        // Never blocks: a full queue drops the event and counts it.
        for e in events {
            out.send(e);
        }
    }
    let totals = state.events.engine.totals();
    let t5 = Instant::now();

    Ok(Detections {
        seq: frame.seq,
        received: frame.received,
        preprocess: t1 - t0,
        inference: t2 - t1,
        postprocess: t3 - t2,
        done: t5,
        boxes,
        tracking: t4 - t3,
        tracks,
        step,
        ids: state.tracker.ids_issued(),
        interval: state.pacing.current(),
        totals,
        event_time: t5 - t4,
    })
}

/// Brings the engine up to date before a run: adopts a newer rule set, recompiles the
/// rules when the frame size changed (they are stored normalised, the engine works in
/// pixels), and zeroes the counts when a reset was requested.
fn sync_rules(ev: &mut EventsState, settings: &DetectorSettings, layout: YuyvLayout) {
    let frame = (layout.width as f32, layout.height as f32);
    let mut changed = ev.engine.frame() != frame;
    if let Some((version, rules)) = settings.rules_if_newer(ev.rules_seen) {
        ev.rules_seen = version;
        ev.rules = rules;
        changed = true;
    }
    // `set_rules` allocates, so only on a change, never on every frame.
    if changed {
        ev.engine.set_rules(&ev.rules, frame.0, frame.1);
    }
    let generation = settings.reset_generation();
    if generation != ev.reset_seen {
        ev.engine.reset_counts();
        ev.reset_seen = generation;
    }
}

/// The preprocessor for `layout`, building it on first use or when the layout changed
/// (its tables are sized for one frame geometry).
fn preprocessor_for(
    pre: &mut Option<Preprocessor>,
    layout: YuyvLayout,
    matrix: ColorMatrix,
) -> &Preprocessor {
    if pre.as_ref().is_some_and(|p| p.layout() != layout) {
        *pre = None;
    }
    pre.get_or_insert_with(|| Preprocessor::new(layout, matrix))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::detect::BBox;
    use crate::events::geometry::Point;
    use crate::track::kalman::BoxVelocity;
    use crate::track::tracker::TrackState;

    #[test]
    fn detector_settings_round_trip() {
        let s = DetectorSettings::default();
        assert_eq!(s.conf(), 0.35);
        s.set_conf(0.42);
        assert_eq!(s.conf(), 0.42);
    }

    #[test]
    fn detector_settings_interval_round_trip() {
        let s = DetectorSettings::default();
        assert_eq!(s.interval(), DetectInterval::EveryFrame);
        s.set_interval(DetectInterval::Auto);
        assert_eq!(s.interval(), DetectInterval::Auto);
        assert_eq!(s.conf(), 0.35);
    }

    #[test]
    fn detector_settings_rules_versioning() {
        let s = DetectorSettings::default();
        assert!(s.rules_if_newer(0).is_none());
        let mut rules = Rules::default();
        rules.add_line(Point::new(0.5, 0.0), Point::new(0.5, 1.0));
        s.set_rules(&rules);
        assert_eq!(s.rules_if_newer(0), Some((1, rules)));
        assert!(s.rules_if_newer(1).is_none());

        assert_eq!(s.reset_generation(), 0);
        s.request_reset();
        assert_eq!(s.reset_generation(), 1);
    }

    /// A Tracked person whose bottom-centre (feet) is at `x`, y 600.
    fn person_at(x: f32, t: Instant) -> TrackView {
        TrackView {
            id: 1,
            class: 0,
            score: 0.9,
            state: TrackState::Tracked,
            bbox: BBox::new(x - 50.0, 300.0, x + 50.0, 600.0),
            velocity: BoxVelocity::default(),
            at: t,
        }
    }

    #[test]
    fn worker_sync_rules_adopts_and_resets() {
        let settings = DetectorSettings::default();
        let mut ev = EventsState::new(None);
        let hd = YuyvLayout::from_len(1920, 1080, 1920 * 1080 * 2).unwrap();

        // No rules set yet: the engine runs with none.
        sync_rules(&mut ev, &settings, hd);
        assert!(ev.engine.totals().rules.is_empty());

        // A new rule set reaches the engine: a vertical line through the middle.
        let mut rules = Rules::default();
        rules.add_line(Point::new(0.5, 0.0), Point::new(0.5, 1.0));
        settings.set_rules(&rules);
        sync_rules(&mut ev, &settings, hd);
        assert_eq!(ev.engine.totals().rules.len(), 1);
        assert_eq!(ev.rules_seen, 1);

        // Walk across it (screen-left is side A): one A→B.
        let t0 = Instant::now();
        let mut out = Vec::new();
        for (i, x) in [800.0, 900.0, 1020.0, 1120.0].into_iter().enumerate() {
            let t = t0 + Duration::from_millis(100 * i as u64);
            ev.engine.update(&[person_at(x, t)], t, i as u64, &mut out);
        }
        assert_eq!(ev.engine.totals().rules[0].a_to_b, 1);

        // A new frame size recompiles (the rule keeps its totals) and is remembered.
        let small = YuyvLayout::from_len(640, 480, 640 * 480 * 2).unwrap();
        sync_rules(&mut ev, &settings, small);
        assert_eq!(ev.engine.frame(), (640.0, 480.0));
        assert_eq!(ev.engine.totals().rules[0].a_to_b, 1);

        // A reset zeroes the totals, once.
        settings.request_reset();
        sync_rules(&mut ev, &settings, small);
        assert_eq!(ev.engine.totals().rules[0].a_to_b, 0);
        assert_eq!(ev.reset_seen, 1);
    }

    #[test]
    fn worker_should_run_paces() {
        let fixed = DetectInterval::Fixed(Duration::from_millis(100));
        let mut pacing = IntervalController::new(fixed);
        let t0 = Instant::now();
        let at = |ms| t0 + Duration::from_millis(ms);
        let runs: Vec<bool> = [0, 33, 66, 100]
            .into_iter()
            .map(|ms| should_run(&mut pacing, fixed, at(ms)))
            .collect();
        assert_eq!(runs, [true, false, false, true]);
        // 33 ms after the last run is too early for Fixed(100 ms), but due once the
        // mode switches to every frame.
        assert!(should_run(&mut pacing, DetectInterval::EveryFrame, at(133)));
    }

    #[test]
    fn detector_rebuilds_preprocessor_on_layout_change() {
        let small = YuyvLayout::from_len(4, 2, 16).unwrap();
        let big = YuyvLayout::from_len(8, 4, 64).unwrap();
        let mut pre = None;
        assert_eq!(
            preprocessor_for(&mut pre, small, ColorMatrix::Bt709).layout(),
            small
        );
        assert_eq!(
            preprocessor_for(&mut pre, small, ColorMatrix::Bt709).layout(),
            small
        );
        assert_eq!(
            preprocessor_for(&mut pre, big, ColorMatrix::Bt709).layout(),
            big
        );
    }
}
