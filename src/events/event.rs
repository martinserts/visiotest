//! The events the rules produce, and the running totals per rule.
//!
//! Plain data: the engine fills them in, the JSON sink prints them and the UI
//! shows the totals. There are four kinds of event:
//!
//! - **`line_cross`**: a track's anchor was confidently on one side of a line and is now
//!   confidently on the other, and the move between went through the drawn segment
//!   (not past its end). It carries the [`Direction`].
//! - **`zone_enter`**: the anchor has been inside a zone for 300 ms in a row. The short
//!   wait (the *settle* time) keeps a box jittering on the edge from entering and
//!   leaving every frame.
//! - **`zone_dwell`**: once per visit, when the time inside reaches the zone's
//!   `dwell_after` ("#7 has been at the desk 10 s").
//! - **`zone_exit`**: the anchor has been outside for 300 ms in a row (reason
//!   [`ExitReason::Left`]), or the track was deleted while inside (reason
//!   [`ExitReason::Lost`]). It carries the total time inside.

use std::time::{Duration, Instant};

use super::rules::RuleId;
use crate::track::TrackId;

/// Which way a line was crossed. A is the positive side of
/// [`super::geometry::signed_distance`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Direction {
    AToB,
    BToA,
}

/// Why a track left a zone.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExitReason {
    /// The anchor moved out of the zone.
    Left,
    /// The track was deleted while inside.
    Lost,
}

/// What happened.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum EventKind {
    LineCross {
        dir: Direction,
    },
    ZoneEnter,
    /// Once per visit, when the time inside reaches the zone's `dwell_after`.
    ZoneDwell {
        dwell: Duration,
    },
    /// `dwell` is the total time inside.
    ZoneExit {
        dwell: Duration,
        reason: ExitReason,
    },
}

/// One event: what happened, to which track, at which rule, and when.
#[derive(Clone, Debug, PartialEq)]
pub struct Event {
    pub kind: EventKind,
    pub rule: RuleId,
    /// The rule's name when the event happened, so a printed line reads on its own.
    pub rule_name: String,
    pub track: TrackId,
    /// COCO class id of the track, an index into `CLASS_NAMES`.
    pub class: u8,
    /// Camera sequence number of the frame that produced it.
    pub seq: u64,
    /// That frame's `Frame::received`.
    pub at: Instant,
}

/// A track inside a zone, for the dwell timers on screen.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Dweller {
    pub track: TrackId,
    /// When it entered (the start of the visit).
    pub since: Instant,
}

/// One rule's running totals (since start or the last reset).
#[derive(Clone, Debug, PartialEq)]
pub struct RuleTotals {
    pub rule: RuleId,
    /// Line crossings A→B; 0 for zones.
    pub a_to_b: u64,
    /// Line crossings B→A; 0 for zones.
    pub b_to_a: u64,
    /// Zone visits (enter events); 0 for lines.
    pub visits: u64,
    /// Zones: the tracks inside now (Lost ones too), by track id. Empty for lines.
    pub inside: Vec<Dweller>,
    /// When this rule last fired, for the UI's flash. It travels with the totals, so it
    /// survives a result the UI never saw.
    pub last_event: Option<Instant>,
}

/// Totals for every rule, in `Rules` order (lines, then zones).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct EventTotals {
    pub rules: Vec<RuleTotals>,
}

impl EventTotals {
    /// The totals of rule `id`, if it exists. A linear search: there are only a few rules.
    pub fn rule(&self, id: RuleId) -> Option<&RuleTotals> {
        self.rules.iter().find(|totals| totals.rule == id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn totals(rule: RuleId, a_to_b: u64) -> RuleTotals {
        RuleTotals {
            rule,
            a_to_b,
            b_to_a: 0,
            visits: 0,
            inside: Vec::new(),
            last_event: None,
        }
    }

    #[test]
    fn events_totals_rule_lookup() {
        let all = EventTotals {
            rules: vec![totals(1, 5), totals(4, 7)],
        };
        assert_eq!(all.rule(1).map(|t| t.a_to_b), Some(5));
        assert_eq!(all.rule(4).map(|t| t.a_to_b), Some(7));
        assert_eq!(all.rule(2), None);
        assert_eq!(EventTotals::default().rule(1), None);
    }
}
