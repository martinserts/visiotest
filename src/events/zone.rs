//! The zone state machine: when has a track entered a zone, stayed, and left it?
//!
//! **Raw inside/outside isn't enough.** Whether the anchor is inside the polygon
//! ([`super::geometry::point_in_polygon`]) is a yes/no per detector run, and near the
//! zone's edge the box's jitter makes it flicker: in, out, in. Reporting every change
//! would spam enter/exit events. So the raw answer is **debounced in time**: a change
//! only counts once it has held for the *settle* time (300 ms by default), like
//! Frigate's `inertia`, but measured in seconds, so it means the same whether the
//! detector runs every frame or every 250 ms.
//!
//! ```text
//! inside?   ─ ─ ─ ███ ─ ███████████████████████████ ─ ─ ─ ─ ─ ─
//!                 ▲ 0.1 s flicker: ignored (< settle)
//!                       ▲ inside for 0.3 s → zone_enter (dwell clock starts at the
//!                         first inside sighting)
//!                                 ▲ dwell_after → zone_dwell (once)
//!                                                   ▲ outside 0.3 s → zone_exit
//! ```
//!
//! **Times are honest.** The debounce delays the *events*, not the times they report:
//! the visit starts at the first inside sighting of the run of sightings that settled,
//! and ends at the first outside sighting. Worked example with settle 0.3 s: the anchor
//! is inside from 2.0 s on, so `zone_enter` fires at 2.3 s with the visit starting at
//! 2.0 s; `zone_dwell` (10 s) fires at 12.0 s; the anchor is outside from 20.0 s on, so
//! `zone_exit` fires at 20.3 s with a dwell of 18.0 s.
//!
//! **Hidden tracks.** A Lost track has no trustworthy position, so its inside/outside
//! state is frozen ([`ZoneMemory::coast`]). It is presumably still there, just hidden,
//! so the dwell clock keeps running. If the tracker deletes it while it is inside, the
//! visit ends with the dwell up to the last time it was actually seen
//! ([`ZoneMemory::gone`]).

use std::time::{Duration, Instant};

use crate::track::TrackId;

/// What a zone rule remembers about one track.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ZoneMemory {
    pub track: TrackId,
    /// The class of the track's latest counted view, for the exit event of a track
    /// that has disappeared from the views.
    pub class: u8,
    /// The debounced state.
    inside: bool,
    /// A raw state different from `inside`, and since when it has held.
    candidate: Option<(bool, Instant)>,
    /// The first inside sighting of this visit; `None` while outside.
    entered_at: Option<Instant>,
    /// Whether this visit's `zone_dwell` has been sent.
    dwell_sent: bool,
    /// The last Tracked observation, for the dwell of a track deleted while inside.
    last_seen: Instant,
    /// Whether the current detector run had a view of this track (the engine's
    /// bookkeeping: memories not seen in a run belong to deleted tracks).
    pub seen: bool,
}

/// A change of a track's zone state, as returned by [`ZoneMemory`]'s methods.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ZoneChange {
    Enter,
    /// The time inside so far, once per visit.
    Dwell(Duration),
    /// The total time inside.
    Exit(Duration),
}

impl ZoneMemory {
    /// A fresh memory for a track first met at `t`: outside.
    pub fn new(track: TrackId, class: u8, t: Instant) -> Self {
        Self {
            track,
            class,
            inside: false,
            candidate: None,
            entered_at: None,
            dwell_sent: false,
            last_seen: t,
            seen: false,
        }
    }

    /// When the current visit started, while the (debounced) state is inside.
    pub fn inside_since(&self) -> Option<Instant> {
        if self.inside { self.entered_at } else { None }
    }

    /// One Tracked sighting at frame time `t`: `raw_inside` says whether the anchor is
    /// inside the polygon right now.
    ///
    /// At most one change per call. When the enter and the dwell fall on the same call
    /// (a `dwell_after` shorter than the time between detector runs), this returns
    /// `Enter` and the next call returns `Dwell`, so the events stay in order.
    pub fn observe(
        &mut self,
        raw_inside: bool,
        t: Instant,
        settle: Duration,
        dwell_after: Duration,
    ) -> Option<ZoneChange> {
        self.last_seen = t;
        if let Some(change) = self.debounce(raw_inside, t, settle) {
            return Some(change);
        }
        self.check_dwell(t, dwell_after)
    }

    /// A Lost view at frame time `t`: no position, so the debounced state is frozen,
    /// but the dwell clock keeps running, so `Dwell` can still fire.
    pub fn coast(&mut self, t: Instant, dwell_after: Duration) -> Option<ZoneChange> {
        self.check_dwell(t, dwell_after)
    }

    /// The track was deleted. If it was inside, returns the visit's dwell, up to the
    /// last time it was seen (not up to the deletion, which comes ~1 s later).
    pub fn gone(&self) -> Option<Duration> {
        let since = self.inside_since()?;
        Some(self.last_seen.saturating_duration_since(since))
    }

    /// Flips the debounced state once the raw state has differed from it for `settle`.
    fn debounce(&mut self, raw_inside: bool, t: Instant, settle: Duration) -> Option<ZoneChange> {
        if raw_inside == self.inside {
            // Back to the settled state before the change held: it was a flicker.
            self.candidate = None;
            return None;
        }
        let since = match self.candidate {
            Some((state, since)) if state == raw_inside => since,
            _ => {
                self.candidate = Some((raw_inside, t));
                t
            }
        };
        if t.saturating_duration_since(since) < settle {
            return None;
        }
        self.candidate = None;
        self.inside = raw_inside;
        if raw_inside {
            // The visit started at the first inside sighting, not now.
            self.entered_at = Some(since);
            self.dwell_sent = false;
            Some(ZoneChange::Enter)
        } else {
            // It ended at the first outside sighting.
            let entered = self.entered_at.take().unwrap_or(since);
            Some(ZoneChange::Exit(since.saturating_duration_since(entered)))
        }
    }

    /// `Dwell` once per visit, when the time inside reaches `dwell_after`.
    fn check_dwell(&mut self, t: Instant, dwell_after: Duration) -> Option<ZoneChange> {
        let since = self.inside_since()?;
        let dwell = t.saturating_duration_since(since);
        if self.dwell_sent || dwell < dwell_after {
            return None;
        }
        self.dwell_sent = true;
        Some(ZoneChange::Dwell(dwell))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SETTLE: Duration = Duration::from_millis(300);

    fn secs(s: f64) -> Duration {
        Duration::from_secs_f64(s)
    }

    /// Observes `inside` every 33 ms over `from..to` scene seconds; returns each change
    /// with the scene time it came at.
    fn play(
        m: &mut ZoneMemory,
        t0: Instant,
        from: f64,
        to: f64,
        inside: bool,
        dwell_after: Duration,
    ) -> Vec<(f64, ZoneChange)> {
        let mut changes = Vec::new();
        let mut s = from;
        while s < to {
            if let Some(c) = m.observe(inside, t0 + secs(s), SETTLE, dwell_after) {
                changes.push((s, c));
            }
            s += 0.033;
        }
        changes
    }

    fn close(a: Duration, b: f64) -> bool {
        (a.as_secs_f64() - b).abs() < 0.05
    }

    #[test]
    fn events_zone_flicker_ignored() {
        let t0 = Instant::now();
        let mut m = ZoneMemory::new(1, 0, t0);
        let dwell = secs(10.0);
        assert_eq!(play(&mut m, t0, 0.0, 1.0, false, dwell), []);
        assert_eq!(play(&mut m, t0, 1.0, 1.1, true, dwell), []);
        assert_eq!(play(&mut m, t0, 1.1, 3.0, false, dwell), []);
        assert_eq!(m.inside_since(), None);
    }

    #[test]
    fn events_zone_enter_dwell_exit() {
        let t0 = Instant::now();
        let mut m = ZoneMemory::new(1, 0, t0);
        let dwell = secs(10.0);
        let inside = play(&mut m, t0, 0.0, 12.0, true, dwell);
        assert_eq!(inside.len(), 2, "{inside:?}");
        let (enter_s, enter) = inside[0];
        assert_eq!(enter, ZoneChange::Enter);
        assert!((enter_s - 0.3).abs() < 0.05, "enter at {enter_s}");
        assert_eq!(
            m.inside_since(),
            Some(t0),
            "the visit starts at the first sighting"
        );
        let (_, ZoneChange::Dwell(d)) = inside[1] else {
            panic!("expected a dwell, got {inside:?}");
        };
        assert!(close(d, 10.0), "{d:?}");

        let outside = play(&mut m, t0, 12.0, 13.0, false, dwell);
        assert_eq!(outside.len(), 1, "{outside:?}");
        let (_, ZoneChange::Exit(total)) = outside[0] else {
            panic!("expected an exit, got {outside:?}");
        };
        assert!(close(total, 12.0), "{total:?}");
        assert_eq!(m.inside_since(), None);
    }

    #[test]
    fn events_zone_coast_keeps_dwell_clock() {
        let t0 = Instant::now();
        let mut m = ZoneMemory::new(1, 0, t0);
        let dwell = secs(2.0);
        let inside = play(&mut m, t0, 0.0, 0.5, true, dwell);
        assert_eq!(inside.len(), 1);
        let last_seen_s = (0..)
            .map(|i| i as f64 * 0.033)
            .take_while(|&s| s < 0.5)
            .last()
            .unwrap();

        // Hidden from here on: only coasting.
        let mut changes = Vec::new();
        for i in 0..100 {
            if let Some(c) = m.coast(t0 + secs(0.5 + i as f64 * 0.033), dwell) {
                changes.push(c);
            }
        }
        assert_eq!(changes.len(), 1, "{changes:?}");
        assert!(matches!(changes[0], ZoneChange::Dwell(d) if close(d, 2.0)));
        let gone = m.gone().expect("it was inside");
        assert!(close(gone, last_seen_s), "{gone:?}");
    }

    #[test]
    fn events_zone_enter_and_dwell_not_same_call() {
        let t0 = Instant::now();
        let mut m = ZoneMemory::new(1, 0, t0);
        let dwell = secs(1.0);
        let at = |s: f64| t0 + secs(s);
        assert_eq!(m.observe(true, at(0.0), SETTLE, dwell), None);
        // 2 s later: settled (enter) and past the dwell time, but only one change.
        assert_eq!(
            m.observe(true, at(2.0), SETTLE, dwell),
            Some(ZoneChange::Enter)
        );
        assert_eq!(
            m.observe(true, at(4.0), SETTLE, dwell),
            Some(ZoneChange::Dwell(secs(4.0)))
        );
        assert_eq!(m.observe(true, at(6.0), SETTLE, dwell), None);
    }
}
