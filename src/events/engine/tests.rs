//! Tests for the event engine, with hand-made views.

use super::*;
use crate::detect::classes::ClassFilter;
use crate::events::rules::Anchor;
use crate::track::kalman::BoxVelocity;

const W: f32 = 1920.0;
const H: f32 = 1080.0;
const PERSON: u8 = 0;
const CAR: u8 = 2;

/// A 100×300 px person centred at (`cx`, `cy`), valid at `at`.
fn view(id: TrackId, cx: f32, cy: f32, state: TrackState, at: Instant) -> TrackView {
    TrackView {
        id,
        class: PERSON,
        score: 0.8,
        state,
        bbox: BBox::from_center(cx, cy, 100.0, 300.0),
        velocity: BoxVelocity::default(),
        at,
    }
}

fn tracked(id: TrackId, cx: f32, at: Instant) -> TrackView {
    view(id, cx, 540.0, TrackState::Tracked, at)
}

fn with_class(v: TrackView, class: u8) -> TrackView {
    TrackView { class, ..v }
}

fn secs(s: f64) -> Duration {
    Duration::from_secs_f64(s)
}

/// A People line at x = `x` (normalised), drawn top → bottom, so A is the screen-left.
fn vertical_line(rules: &mut Rules, x: f32) -> RuleId {
    rules.add_line(Point::new(x, 0.0), Point::new(x, 1.0))
}

/// A People zone over the middle half of the frame (x 480–1440 px), anchored at the centre.
fn middle_zone(rules: &mut Rules) -> RuleId {
    let id = rules.add_zone(vec![
        Point::new(0.25, 0.0),
        Point::new(0.75, 0.0),
        Point::new(0.75, 1.0),
        Point::new(0.25, 1.0),
    ]);
    rules.zone_mut(id).unwrap().anchor = Anchor::Center;
    id
}

fn engine_with(rules: &Rules) -> EventEngine {
    let mut engine = EventEngine::new(EngineConfig::default());
    engine.set_rules(rules, W, H);
    engine
}

/// One run; returns its events.
fn run(engine: &mut EventEngine, views: &[TrackView], t: Instant) -> Vec<Event> {
    let mut out = Vec::new();
    engine.update(views, t, 0, &mut out);
    out
}

fn kinds(events: &[Event]) -> Vec<EventKind> {
    events.iter().map(|e| e.kind).collect()
}

const A_TO_B: EventKind = EventKind::LineCross {
    dir: Direction::AToB,
};
const B_TO_A: EventKind = EventKind::LineCross {
    dir: Direction::BToA,
};

#[test]
fn events_engine_lost_freezes_then_recovers_across() {
    let mut rules = Rules::default();
    let line = vertical_line(&mut rules, 0.5);
    let mut engine = engine_with(&rules);
    let t0 = Instant::now();
    let lost = |cx: f32, s: f64| view(1, cx, 540.0, TrackState::Lost, t0 + secs(s));

    assert_eq!(run(&mut engine, &[tracked(1, 900.0, t0)], t0), []);
    // Hidden: the guessed boxes walk over the line, but a guess never moves the side.
    assert_eq!(run(&mut engine, &[lost(960.0, 0.1)], t0 + secs(0.1)), []);
    assert_eq!(run(&mut engine, &[lost(1100.0, 0.2)], t0 + secs(0.2)), []);
    let t = t0 + secs(0.3);
    let events = run(&mut engine, &[tracked(1, 1020.0, t)], t);
    assert_eq!(kinds(&events), [A_TO_B], "counted on recovery");
    assert_eq!((events[0].rule, events[0].track), (line, 1));
    assert_eq!(engine.totals().rule(line).unwrap().a_to_b, 1);
}

#[test]
fn events_engine_gone_inside_zone_exits_lost() {
    let mut rules = Rules::default();
    let zone = middle_zone(&mut rules);
    let mut engine = engine_with(&rules);
    let t0 = Instant::now();
    let mut events = Vec::new();
    for i in 0..=5 {
        let t = t0 + secs(i as f64 * 0.1);
        events.extend(run(&mut engine, &[tracked(4, 960.0, t)], t));
    }
    assert_eq!(kinds(&events), [EventKind::ZoneEnter]);
    let totals = engine.totals();
    let inside = &totals.rule(zone).unwrap().inside;
    assert_eq!(
        inside,
        &[Dweller {
            track: 4,
            since: t0
        }]
    );

    // The track is deleted while inside: it exits, with the dwell up to its last sighting.
    let events = run(&mut engine, &[], t0 + secs(1.5));
    assert_eq!(events.len(), 1);
    let EventKind::ZoneExit { dwell, reason } = events[0].kind else {
        panic!("expected an exit, got {events:?}");
    };
    assert_eq!(reason, ExitReason::Lost);
    assert_eq!(dwell, secs(0.5));
    assert_eq!((events[0].track, events[0].class), (4, PERSON));
    assert!(engine.totals().rule(zone).unwrap().inside.is_empty());
}

#[test]
fn events_engine_class_filter_and_flicker() {
    let mut rules = Rules::default();
    let line = vertical_line(&mut rules, 0.5);
    let zone = middle_zone(&mut rules);
    let mut engine = engine_with(&rules);
    let t0 = Instant::now();
    let at = |s: f64| t0 + secs(s);

    // A car crossing a People line: nothing.
    let car = |cx: f32, s: f64| with_class(tracked(1, cx, at(s)), CAR);
    assert_eq!(run(&mut engine, &[car(900.0, 0.0)], at(0.0)), []);
    assert_eq!(run(&mut engine, &[car(1020.0, 0.1)], at(0.1)), []);

    // A person on side A and inside the zone for a while.
    let mut events = Vec::new();
    for i in 0..=5 {
        let s = 1.0 + i as f64 * 0.1;
        events.extend(run(&mut engine, &[tracked(2, 900.0, at(s))], at(s)));
    }
    assert_eq!(kinds(&events), [EventKind::ZoneEnter]);
    // One run labelled "car": neither rule counts it, but it isn't gone either.
    let flicker = with_class(tracked(2, 940.0, at(1.6)), CAR);
    assert_eq!(run(&mut engine, &[flicker], at(1.6)), [], "no lost-exit");
    assert_eq!(engine.totals().rule(zone).unwrap().inside.len(), 1);
    // A person again, now on side B: the line still remembered side A.
    let events = run(&mut engine, &[tracked(2, 1020.0, at(1.7))], at(1.7));
    assert_eq!(kinds(&events), [A_TO_B]);
    assert_eq!(events[0].rule, line);
}

#[test]
fn events_engine_set_rules_keeps_and_resets() {
    let mut rules = Rules::default();
    let line = vertical_line(&mut rules, 0.5);
    // A second line, far from the walker, to remove at the end.
    let other = vertical_line(&mut rules, 0.1);
    let mut engine = engine_with(&rules);
    let t0 = Instant::now();
    let at = |s: f64| t0 + secs(s);

    // Unchanged rules: no crossing is lost across `set_rules`.
    assert_eq!(run(&mut engine, &[tracked(1, 900.0, at(0.0))], at(0.0)), []);
    engine.set_rules(&rules, W, H);
    let events = run(&mut engine, &[tracked(1, 1020.0, at(0.1))], at(0.1));
    assert_eq!(kinds(&events), [A_TO_B]);

    // Renamed: the memory is kept, and events carry the new name.
    rules.line_mut(line).unwrap().name = "door".to_string();
    engine.set_rules(&rules, W, H);
    let events = run(&mut engine, &[tracked(1, 900.0, at(0.2))], at(0.2));
    assert_eq!(kinds(&events), [B_TO_A]);
    assert_eq!(events[0].rule_name, "door");

    // Moved to x 940.8 px, between the last confident point (900) and the next (960).
    // Remembering side A would count a crossing that is really the line moving.
    rules.line_mut(line).unwrap().a.x = 0.49;
    rules.line_mut(line).unwrap().b.x = 0.49;
    engine.set_rules(&rules, W, H);
    assert_eq!(run(&mut engine, &[tracked(1, 960.0, at(0.3))], at(0.3)), []);
    let totals = engine.totals();
    let kept = totals.rule(line).unwrap();
    assert_eq!(
        (kept.a_to_b, kept.b_to_a),
        (1, 1),
        "totals survive the move"
    );

    // Removed: gone from the totals.
    assert!(rules.remove(other));
    engine.set_rules(&rules, W, H);
    let totals = engine.totals();
    assert_eq!(totals.rules.len(), 1);
    assert_eq!(totals.rule(other), None);
}

#[test]
fn events_engine_reset_counts() {
    let mut rules = Rules::default();
    let line = vertical_line(&mut rules, 0.5);
    let zone = middle_zone(&mut rules);
    let mut engine = engine_with(&rules);
    let t0 = Instant::now();
    for (i, cx) in [900.0, 920.0, 1020.0, 1030.0, 1040.0, 1050.0]
        .into_iter()
        .enumerate()
    {
        let t = t0 + secs(i as f64 * 0.1);
        run(&mut engine, &[tracked(1, cx, t)], t);
    }
    let totals = engine.totals();
    assert_eq!(totals.rule(line).unwrap().a_to_b, 1);
    assert_eq!(totals.rule(zone).unwrap().visits, 1);

    engine.reset_counts();
    let totals = engine.totals();
    let (l, z) = (totals.rule(line).unwrap(), totals.rule(zone).unwrap());
    assert_eq!((l.a_to_b, l.b_to_a, l.last_event), (0, 0, None));
    assert_eq!((z.visits, z.last_event), (0, None));
    assert_eq!(z.inside.len(), 1, "still inside");
}

#[test]
fn events_engine_event_order() {
    let mut rules = Rules::default();
    let first = vertical_line(&mut rules, 0.5);
    let second = vertical_line(&mut rules, 0.51);
    // Both rules count every class, and track 2 is a car: class filters don't reorder.
    rules.line_mut(second).unwrap().classes = ClassFilter::all();
    rules.line_mut(first).unwrap().classes = ClassFilter::all();
    let mut engine = engine_with(&rules);
    let t0 = Instant::now();
    let t1 = t0 + secs(0.1);
    let before = [
        tracked(1, 900.0, t0),
        with_class(tracked(2, 880.0, t0), CAR),
    ];
    assert_eq!(run(&mut engine, &before, t0), []);
    let after = [
        tracked(1, 1100.0, t1),
        with_class(tracked(2, 1080.0, t1), CAR),
    ];
    let events = run(&mut engine, &after, t1);
    let order: Vec<(RuleId, TrackId)> = events.iter().map(|e| (e.rule, e.track)).collect();
    assert_eq!(order, [(first, 1), (first, 2), (second, 1), (second, 2)]);
    assert_eq!(events[1].class, CAR);
}
