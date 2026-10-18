//! The event engine on the tracker's synthetic scenes, through the real tracker.
//!
//! The unit tests feed the engine hand-made views. Here the views come from the real
//! [`Tracker`], fed by the simulated detector ([`crate::track::sim`]): jittered boxes,
//! half-boxes behind a pole, tracks that go Lost and are found again. The scenes know
//! who walks where, so the right counts are known: every walker moves horizontally
//! through the frame centre, and a vertical line at x = 0.5 drawn top → bottom (A on
//! the left) is crossed A→B by a walker going right and B→A by one going left.
//!
//! [`play`] copies `sim::run_with`'s loop, and runs the engine **only on detector runs**,
//! as the detector thread does: the counts must match what happens live.

use std::time::{Duration, Instant};

use super::engine::{EngineConfig, EventEngine};
use super::event::{Event, EventKind, EventTotals, ExitReason};
use super::geometry::Point;
use super::rules::{Anchor, RuleId, Rules};
use crate::detect::classes::{ClassFilter, Group};
use crate::detect::interval::{DetectInterval, IntervalController};
use crate::track::sim::{FPS, FRAME_H, FRAME_W, Rng, Scene, scenes};
use crate::track::tracker::{Tracker, TrackerConfig};

/// The confidence slider's default: the tracker's high threshold.
const HIGH_CONF: f32 = 0.35;

/// The pacing modes the report compares (the same as the tracker's `sim_report`).
const MODES: [DetectInterval; 4] = [
    DetectInterval::EveryFrame,
    DetectInterval::Fixed(Duration::from_millis(100)),
    DetectInterval::Fixed(Duration::from_millis(250)),
    DetectInterval::Auto,
];

/// What playing a scene produced.
struct Played {
    totals: EventTotals,
    events: Vec<Event>,
}

fn scene_named(name: &str) -> Scene {
    scenes()
        .into_iter()
        .find(|s| s.name == name)
        .unwrap_or_else(|| panic!("no scene {name}"))
}

/// Plays scene `name` at 30 fps, paced by `mode`, through a fresh tracker and an engine
/// running `rules` on a 1920×1080 frame.
fn play(name: &str, mode: DetectInterval, rules: &Rules) -> Played {
    let scene = scene_named(name);
    let t0 = Instant::now();
    let mut rng = Rng::new(scene.seed);
    let mut tracker = Tracker::new(TrackerConfig::default());
    let mut controller = IntervalController::new(mode);
    let mut engine = EventEngine::new(EngineConfig::default());
    engine.set_rules(rules, FRAME_W, FRAME_H);
    let (mut dets, mut views, mut run_events, mut events) =
        (Vec::new(), Vec::new(), Vec::new(), Vec::new());

    for i in 0..scene.frames() {
        // From the index, not by adding 33 ms per frame, so no rounding drift adds up.
        let secs = i as f64 / FPS;
        let t = t0 + Duration::from_secs_f64(secs);
        if !controller.is_due(t) {
            // A skipped frame: no detector run, so no tracker and no engine either.
            continue;
        }
        controller.ran(t);
        scene.detections(secs as f32, &mut rng, &mut dets);
        let step = tracker.update(&dets, HIGH_CONF, t);
        controller.on_result(step.interval_signal());
        tracker.views(&mut views);
        engine.update(&views, t, u64::from(i), &mut run_events);
        events.extend_from_slice(&run_events);
    }
    Played {
        totals: engine.totals(),
        events,
    }
}

/// Rules with one vertical line at x = `x` (normalised), drawn top → bottom (A = left).
fn line_rules(x: f32, classes: ClassFilter) -> (Rules, RuleId) {
    let mut rules = Rules::default();
    let id = rules.add_line(Point::new(x, 0.0), Point::new(x, 1.0));
    rules.line_mut(id).unwrap().classes = classes;
    (rules, id)
}

/// Rules with one rectangular zone `x0..x1` × `y0..y1` (normalised), centre anchor,
/// a 1 s dwell time.
fn zone_rules(x0: f32, x1: f32, y0: f32, y1: f32, classes: ClassFilter) -> (Rules, RuleId) {
    let mut rules = Rules::default();
    let corners = vec![
        Point::new(x0, y0),
        Point::new(x1, y0),
        Point::new(x1, y1),
        Point::new(x0, y1),
    ];
    let id = rules.add_zone(corners);
    let zone = rules.zone_mut(id).unwrap();
    zone.anchor = Anchor::Center;
    zone.classes = classes;
    zone.dwell_after = Duration::from_secs(1);
    (rules, id)
}

/// The rules' default class filter.
fn people() -> ClassFilter {
    ClassFilter::only(Group::People)
}

/// The frame's centre line, counting people.
fn centre_line() -> (Rules, RuleId) {
    line_rules(0.5, people())
}

/// The crossings of line `id` as (A→B, B→A).
fn crossings(played: &Played, id: RuleId) -> (u64, u64) {
    let totals = played.totals.rule(id).expect("the rule has totals");
    (totals.a_to_b, totals.b_to_a)
}

/// The kinds of the events, in order.
fn kinds(played: &Played) -> Vec<EventKind> {
    played.events.iter().map(|e| e.kind).collect()
}

/// Exactly one enter, one dwell and (if `exits`) one exit, in that order, with exit
/// reason Left.
fn assert_one_visit(played: &Played, exits: bool) {
    let kinds = kinds(played);
    let expected_len = if exits { 3 } else { 2 };
    assert_eq!(kinds.len(), expected_len, "{kinds:?}");
    assert_eq!(kinds[0], EventKind::ZoneEnter);
    assert!(matches!(kinds[1], EventKind::ZoneDwell { .. }), "{kinds:?}");
    if exits {
        assert!(
            matches!(
                kinds[2],
                EventKind::ZoneExit {
                    reason: ExitReason::Left,
                    ..
                }
            ),
            "{kinds:?}"
        );
    }
}

#[test]
fn events_sim_walk() {
    let (line, line_id) = centre_line();
    let (zone, _) = zone_rules(0.25, 0.75, 0.0, 1.0, people());
    for mode in [DetectInterval::EveryFrame, DetectInterval::Auto] {
        let played = play("walk", mode, &line);
        assert_eq!(crossings(&played, line_id), (1, 0), "{mode}");
        assert_one_visit(&play("walk", mode, &zone), true);
    }
}

#[test]
fn events_sim_pole_counts_on_recovery() {
    // The walker is hidden behind the pole right at the line; its track goes Lost there
    // and is counted when it is found again on the other side.
    let (line, id) = centre_line();
    let played = play("pole", DetectInterval::EveryFrame, &line);
    assert_eq!(crossings(&played, id), (1, 0));
}

#[test]
fn events_sim_crossing_both_ways() {
    let (line, id) = centre_line();
    let played = play("crossing", DetectInterval::EveryFrame, &line);
    assert_eq!(crossings(&played, id), (1, 1));
}

#[test]
fn events_sim_still_cups() {
    // Cups at x 500, 900 and 1300 that never move: a line between them counts nothing.
    let (line, _) = line_rules(0.5, ClassFilter::all());
    let played = play("still", DetectInterval::EveryFrame, &line);
    assert_eq!(kinds(&played), []);

    // The zone: x 768–1152, y 540–1080, so it holds only the cup at x 900.
    let (zone, id) = zone_rules(0.4, 0.6, 0.5, 1.0, ClassFilter::all());
    let played = play("still", DetectInterval::EveryFrame, &zone);
    assert_one_visit(&played, false);
    assert_eq!(played.totals.rule(id).unwrap().inside.len(), 1);
}

#[test]
fn events_sim_stop_near_line() {
    // The walker stands at x 910 for 1.5 s, 10 px past a line at x 900: inside the dead
    // band at times. It crossed once each way, and the jitter adds nothing.
    let (line, id) = line_rules(900.0 / FRAME_W, people());
    let played = play("stop", DetectInterval::EveryFrame, &line);
    assert_eq!(crossings(&played, id), (1, 1));
}

/// The true crossings of the vertical line at x `line_x` px, as (A→B, B→A): each time
/// a walker's centre passes it within the walker's time window and the scene.
///
/// The line runs through the middle of the frame, so a walker crossing it is always in
/// the frame. One hidden behind an occluder there still crossed it: `pole` expects that
/// crossing to be counted when the track is found again.
fn expected_crossings(scene: &Scene, line_x: f32) -> (u32, u32) {
    let (mut a_to_b, mut b_to_a) = (0, 0);
    for w in &scene.walkers {
        let vx = w.velocity.0;
        if vx == 0.0 {
            continue;
        }
        // When the centre is at the line.
        let s = (line_x - w.start.0) / vx;
        if s < w.enter_s || s >= w.leave_s.min(scene.duration_s) {
            continue;
        }
        if vx > 0.0 {
            a_to_b += 1;
        } else {
            b_to_a += 1;
        }
    }
    (a_to_b, b_to_a)
}

/// Run with `cargo test --release events_sim_report -- --ignored --nocapture`.
#[test]
#[ignore = "a report, not a check: run it with --ignored --nocapture"]
fn events_sim_report() {
    let (line, id) = line_rules(0.5, ClassFilter::all());
    println!();
    println!("Line at x 0.5 (all classes): counted vs expected crossings, per scene and mode");
    println!(
        "{:<9} {:<7} {:>5} {:>5}   {:>9}",
        "scene", "mode", "A→B", "B→A", "expected"
    );
    for scene in scenes() {
        let (want_ab, want_ba) = expected_crossings(&scene, 0.5 * FRAME_W);
        for mode in MODES {
            let played = play(scene.name, mode, &line);
            let (ab, ba) = crossings(&played, id);
            println!(
                "{:<9} {:<7} {ab:>5} {ba:>5}   {:>9}",
                scene.name,
                mode.to_string(),
                format!("{want_ab} / {want_ba}"),
            );
        }
    }
    println!();
    println!(
        "Note: `wall` hides the walker for ~1.8 s, longer than a Lost track lives (lost_for 1 s), \
         so the track is deleted behind the wall and a new one starts on the other side: \
         the identity and the crossing are lost."
    );
    println!(
        "Note: Fixed 250 ms never confirms a fast walker (a new track starts \
         with zero velocity and the walker has left its box by the next run), so nothing \
         is counted for it."
    );
}
