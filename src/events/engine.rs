//! The event engine: tracks in, events out, plus running totals per rule.
//!
//! It runs on the detector thread right after the tracker, once per detector run, on
//! each track's Kalman estimate at the frame's time. It holds the rules converted to
//! frame pixels and, per rule, a small memory per track (a [`LineMemory`] or a
//! [`ZoneMemory`]). One run:
//!
//! ```text
//! for each rule (lines, then zones):
//!     for each view (by id) whose class the rule counts:
//!         Tracked → observe (may emit)        Lost → freeze (zones: coast)
//!     memories of tracks no longer in the views → gone (zones: exit, reason lost)
//! ```
//!
//! The per-track bookkeeping that lines and zones share (find or make a track's
//! memory, the label-flicker rule) is `memory_for`.
//!
//! **Only confirmed tracks count.** The tracker's views hold Tracked and Lost tracks,
//! never Tentative ones, so a one-frame false positive never crosses anything. A Lost
//! track's box is a guess, so it never moves rule state: a walker hidden behind a pole
//! at the line is counted when it is found again on the other side. A track missing
//! from the views has been deleted.
//!
//! **The dead band scales with the box.** The line's margin is
//! `max(min_margin_px, margin_frac × min(box w, box h))`: the anchor's jitter grows with
//! the box (a person near the camera is big, and so are their box's wobbles), so a
//! fixed pixel band would be too small up close and too big far away.
//!
//! **Rule changes** ([`EventEngine::set_rules`]) never produce events: a rule that
//! moved keeps its counts and forgets which side each track was on, so moving a line
//! over someone doesn't count them.

#[cfg(test)]
mod tests;

use std::time::{Duration, Instant};

use super::event::{Direction, Dweller, Event, EventKind, EventTotals, ExitReason, RuleTotals};
use super::geometry::{Point, anchor_point, point_in_polygon, to_pixels};
use super::line::{LineMemory, PixelLine};
use super::rules::{LineRule, RuleId, Rules, ZoneRule};
use super::zone::{ZoneChange, ZoneMemory};
use crate::detect::BBox;
use crate::detect::classes::ClassFilter;
use crate::track::TrackId;
use crate::track::tracker::{TrackState, TrackView};

/// The engine's tuning.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EngineConfig {
    /// Dead band as a fraction of the box's smaller side: 10 % of a 100 px wide person
    /// is 10 px, about three times the sim's box jitter.
    pub margin_frac: f32,
    /// The dead band's floor, for small (far away) boxes: a few pixels of jitter remain
    /// however small the box.
    pub min_margin_px: f32,
    /// Zone debounce: a change of inside/outside counts once it has held this long.
    pub settle: Duration,
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            margin_frac: 0.10,
            min_margin_px: 4.0,
            // Like Frigate's `inertia` (3 frames at 10 fps), but in time, so it means the
            // same at every detect interval.
            settle: Duration::from_millis(300),
        }
    }
}

/// A line rule ready to run: its source rule, the line in pixels and its state.
struct CompiledLine {
    /// The rule as given (normalised), to name events and to spot changes.
    rule: LineRule,
    line: PixelLine,
    /// Sorted by track id.
    memory: Vec<LineMemory>,
    totals: RuleTotals,
}

/// A zone rule ready to run: its source rule, the polygon in pixels and its state.
struct CompiledZone {
    rule: ZoneRule,
    polygon: Vec<Point>,
    /// Sorted by track id.
    memory: Vec<ZoneMemory>,
    totals: RuleTotals,
}

/// Turns the tracker's views into events, rule by rule. See the module doc.
pub struct EventEngine {
    cfg: EngineConfig,
    lines: Vec<CompiledLine>,
    zones: Vec<CompiledZone>,
    /// The frame size the rules were compiled for, px.
    frame: (f32, f32),
}

impl EventEngine {
    pub fn new(cfg: EngineConfig) -> Self {
        Self {
            cfg,
            lines: Vec::new(),
            zones: Vec::new(),
            frame: (0.0, 0.0),
        }
    }

    /// Adopt `rules` for frames `frame_w × frame_h` px. A rule whose id and geometry are
    /// unchanged keeps its totals and track memory. A changed rule keeps its totals and
    /// forgets its track memory (no event comes from the change). A removed rule is dropped.
    ///
    /// "Geometry" here is the points, the anchor and the class filter; a new name or
    /// dwell time keeps the memory. A new frame size forgets every memory, since the
    /// remembered pixel positions no longer match.
    pub fn set_rules(&mut self, rules: &Rules, frame_w: f32, frame_h: f32) {
        let resized = self.frame != (frame_w, frame_h);
        self.frame = (frame_w, frame_h);

        let mut old_lines = std::mem::take(&mut self.lines);
        for rule in &rules.lines {
            let old = take_by_id(&mut old_lines, rule.id);
            self.lines
                .push(compile_line(rule, frame_w, frame_h, old, resized));
        }

        let mut old_zones = std::mem::take(&mut self.zones);
        for rule in &rules.zones {
            let old = take_by_id(&mut old_zones, rule.id);
            self.zones
                .push(compile_zone(rule, frame_w, frame_h, old, resized));
        }
    }

    /// Totals to zero; `inside` is kept (they are still there), and so is every track's
    /// memory, so nobody is counted again.
    pub fn reset_counts(&mut self) {
        let lines = self.lines.iter_mut().map(|c| &mut c.totals);
        let zones = self.zones.iter_mut().map(|c| &mut c.totals);
        for totals in lines.chain(zones) {
            *totals = RuleTotals {
                inside: std::mem::take(&mut totals.inside),
                ..empty_totals(totals.rule)
            };
        }
    }

    /// The frame size the rules were last compiled for, px (0 × 0 before the first).
    pub fn frame(&self) -> (f32, f32) {
        self.frame
    }

    /// One detector run: the tracker's views (by id, Tracked + Lost) at frame time `t`.
    /// Clears `out`, then pushes this run's events: rules in order, tracks by id within a rule.
    pub fn update(&mut self, views: &[TrackView], t: Instant, seq: u64, out: &mut Vec<Event>) {
        out.clear();
        for line in &mut self.lines {
            line.update(&self.cfg, views, t, seq, out);
        }
        for zone in &mut self.zones {
            zone.update(&self.cfg, views, t, seq, out);
        }
    }

    /// Every rule's totals, in `Rules` order (lines, then zones).
    pub fn totals(&self) -> EventTotals {
        let lines = self.lines.iter().map(|c| c.totals.clone());
        let zones = self.zones.iter().map(|c| c.totals.clone());
        EventTotals {
            rules: lines.chain(zones).collect(),
        }
    }
}

/// A compiled rule, as `set_rules` needs it: which rule it came from.
trait CompiledRule {
    fn id(&self) -> RuleId;
}

impl CompiledRule for CompiledLine {
    fn id(&self) -> RuleId {
        self.rule.id
    }
}

impl CompiledRule for CompiledZone {
    fn id(&self) -> RuleId {
        self.rule.id
    }
}

/// Takes the compiled rule with this `id` out of `old`, if there is one. The order of
/// `old` doesn't matter (it is thrown away afterwards), so `swap_remove` is enough.
fn take_by_id<T: CompiledRule>(old: &mut Vec<T>, id: RuleId) -> Option<T> {
    let i = old.iter().position(|c| c.id() == id)?;
    Some(old.swap_remove(i))
}

fn compile_line(
    rule: &LineRule,
    w: f32,
    h: f32,
    old: Option<CompiledLine>,
    resized: bool,
) -> CompiledLine {
    let line = PixelLine {
        a: to_pixels(rule.a, w, h),
        b: to_pixels(rule.b, w, h),
    };
    let Some(old) = old else {
        return CompiledLine {
            rule: rule.clone(),
            line,
            memory: Vec::new(),
            totals: empty_totals(rule.id),
        };
    };
    let same = !resized
        && old.rule.a == rule.a
        && old.rule.b == rule.b
        && old.rule.anchor == rule.anchor
        && old.rule.classes == rule.classes;
    let mut memory = old.memory;
    if !same {
        memory.clear();
    }
    CompiledLine {
        rule: rule.clone(),
        line,
        memory,
        totals: old.totals,
    }
}

fn compile_zone(
    rule: &ZoneRule,
    w: f32,
    h: f32,
    old: Option<CompiledZone>,
    resized: bool,
) -> CompiledZone {
    let polygon = rule.points.iter().map(|&p| to_pixels(p, w, h)).collect();
    let Some(old) = old else {
        return CompiledZone {
            rule: rule.clone(),
            polygon,
            memory: Vec::new(),
            totals: empty_totals(rule.id),
        };
    };
    let same = !resized
        && old.rule.points == rule.points
        && old.rule.anchor == rule.anchor
        && old.rule.classes == rule.classes;
    let (mut memory, mut totals) = (old.memory, old.totals);
    if !same {
        // Forgotten tracks are no longer inside, as far as this rule knows.
        memory.clear();
        totals.inside.clear();
    }
    CompiledZone {
        rule: rule.clone(),
        polygon,
        memory,
        totals,
    }
}

fn empty_totals(rule: RuleId) -> RuleTotals {
    RuleTotals {
        rule,
        a_to_b: 0,
        b_to_a: 0,
        visits: 0,
        inside: Vec::new(),
        last_event: None,
    }
}

/// The line's dead band for one box, px (see the module doc).
fn dead_band(cfg: &EngineConfig, b: BBox) -> f32 {
    cfg.min_margin_px
        .max(cfg.margin_frac * b.width().min(b.height()))
}

/// What the engine needs from a per-track memory ([`LineMemory`], [`ZoneMemory`]) to
/// keep its list in step with the tracker's views.
trait TrackMemory {
    fn track(&self) -> TrackId;
    fn set_seen(&mut self, seen: bool);
}

impl TrackMemory for LineMemory {
    fn track(&self) -> TrackId {
        self.track
    }

    fn set_seen(&mut self, seen: bool) {
        self.seen = seen;
    }
}

impl TrackMemory for ZoneMemory {
    fn track(&self) -> TrackId {
        self.track
    }

    fn set_seen(&mut self, seen: bool) {
        self.seen = seen;
    }
}

/// Starts a run: no memory has been seen yet. The ones still unseen after the views
/// belong to tracks the tracker deleted.
fn begin_run<M: TrackMemory>(memory: &mut [M]) {
    for m in memory {
        m.set_seen(false);
    }
}

/// This run's memory for `view`, marked seen: found by binary search, or made with
/// `make` and inserted where the search says it belongs. `None` when `classes` filters
/// the view out.
///
/// The list is kept sorted by track id: a binary search finds a track, and its `Err` is
/// where a new one goes. There are tens of tracks at most, it allocates only while it
/// grows, and the memories stay in id order for free, so no `HashMap` is needed.
///
/// A filtered-out view still marks an existing memory seen. Otherwise a one-run label
/// flicker (person → car) would look like a deleted track: a line would forget its
/// side, and a zone would fire a lost-exit.
fn memory_for<'a, M: TrackMemory>(
    memory: &'a mut Vec<M>,
    view: &TrackView,
    classes: ClassFilter,
    make: impl FnOnce() -> M,
) -> Option<&'a mut M> {
    let found = memory.binary_search_by_key(&view.id, |m| m.track());
    if !classes.contains(view.class) {
        if let Ok(i) = found {
            memory[i].set_seen(true);
        }
        return None;
    }
    // `make` runs only for a new track, so a known one costs no allocation.
    let i = found.unwrap_or_else(|i| {
        memory.insert(i, make());
        i
    });
    let m = &mut memory[i];
    m.set_seen(true);
    Some(m)
}

impl CompiledLine {
    fn update(
        &mut self,
        cfg: &EngineConfig,
        views: &[TrackView],
        t: Instant,
        seq: u64,
        out: &mut Vec<Event>,
    ) {
        begin_run(&mut self.memory);
        for view in views {
            let make = || LineMemory::new(view.id);
            let Some(memory) = memory_for(&mut self.memory, view, self.rule.classes, make) else {
                continue;
            };
            if view.state != TrackState::Tracked {
                // Lost: the box is a guess, so the side freezes.
                continue;
            }
            let p = anchor_point(view.bbox, self.rule.anchor);
            let Some(dir) = memory.observe(&self.line, p, dead_band(cfg, view.bbox)) else {
                continue;
            };
            match dir {
                Direction::AToB => self.totals.a_to_b += 1,
                Direction::BToA => self.totals.b_to_a += 1,
            }
            self.totals.last_event = Some(t);
            out.push(Event {
                kind: EventKind::LineCross { dir },
                rule: self.rule.id,
                rule_name: self.rule.name.clone(),
                track: view.id,
                class: view.class,
                seq,
                at: t,
            });
        }
        // A deleted track was on some side of the line; nothing to report.
        self.memory.retain(|m| m.seen);
    }
}

impl CompiledZone {
    fn update(
        &mut self,
        cfg: &EngineConfig,
        views: &[TrackView],
        t: Instant,
        seq: u64,
        out: &mut Vec<Event>,
    ) {
        let first_event = out.len();
        begin_run(&mut self.memory);
        for view in views {
            let make = || ZoneMemory::new(view.id, view.class, t);
            let Some(memory) = memory_for(&mut self.memory, view, self.rule.classes, make) else {
                continue;
            };
            memory.class = view.class;
            let change = if view.state == TrackState::Tracked {
                let p = anchor_point(view.bbox, self.rule.anchor);
                let inside = point_in_polygon(p, &self.polygon);
                memory.observe(inside, t, cfg.settle, self.rule.dwell_after)
            } else {
                // Lost: inside/outside freezes, the dwell clock runs on.
                memory.coast(t, self.rule.dwell_after)
            };
            if let Some(change) = change {
                let kind = match change {
                    ZoneChange::Enter => EventKind::ZoneEnter,
                    ZoneChange::Dwell(dwell) => EventKind::ZoneDwell { dwell },
                    ZoneChange::Exit(dwell) => EventKind::ZoneExit {
                        dwell,
                        reason: ExitReason::Left,
                    },
                };
                self.push(kind, view.id, view.class, t, seq, out);
            }
        }

        // Tracks that have left the views were deleted by the tracker.
        for i in 0..self.memory.len() {
            let memory = self.memory[i];
            if memory.seen {
                continue;
            }
            if let Some(dwell) = memory.gone() {
                let reason = ExitReason::Lost;
                let kind = EventKind::ZoneExit { dwell, reason };
                self.push(kind, memory.track, memory.class, t, seq, out);
            }
        }
        self.memory.retain(|m| m.seen);
        // The lost-exits came after the views' events; put this rule's events back in
        // track order. Each track has at most one event per rule and run, so the keys
        // are unique and the (allocation-free) unstable sort is enough.
        out[first_event..].sort_unstable_by_key(|e| e.track);

        self.totals.inside.clear();
        self.totals
            .inside
            .extend(self.memory.iter().filter_map(|m| {
                let since = m.inside_since()?;
                Some(Dweller {
                    track: m.track,
                    since,
                })
            }));
    }

    /// Pushes one event of this zone and counts it in the totals.
    fn push(
        &mut self,
        kind: EventKind,
        track: TrackId,
        class: u8,
        t: Instant,
        seq: u64,
        out: &mut Vec<Event>,
    ) {
        if kind == EventKind::ZoneEnter {
            self.totals.visits += 1;
        }
        self.totals.last_event = Some(t);
        out.push(Event {
            kind,
            rule: self.rule.id,
            rule_name: self.rule.name.clone(),
            track,
            class,
            seq,
            at: t,
        });
    }
}
