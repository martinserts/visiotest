//! One event → one line of JSON.
//!
//! The stream is in the **JSON Lines** format: every event is one complete JSON object
//! on its own line. A consumer can pipe it into `jq`, `tail -f` it or read it line by
//! line from another program, and it never sees half an event, because a line is only
//! ever written whole. The four kinds look like this:
//!
//! ```json
//! {"type":"line_cross","t":1760735400123,"seq":4512,"rule":1,"rule_name":"door","track":7,"class":"person","dir":"a_to_b"}
//! {"type":"zone_enter","t":…,"seq":…,"rule":2,"rule_name":"desk","track":7,"class":"person"}
//! {"type":"zone_dwell","t":…,"seq":…,"rule":2,"rule_name":"desk","track":7,"class":"person","dwell_s":10.0}
//! {"type":"zone_exit","t":…,"seq":…,"rule":2,"rule_name":"desk","track":7,"class":"person","dwell_s":42.3,"reason":"left"}
//! ```
//!
//! **`t` is when the frame arrived** (`Frame::received`, as Unix milliseconds), not when
//! the line was printed. The detector needs 15–40 ms per frame and may run every
//! 500 ms, so the print time would shift by a varying amount; the frame's time is when
//! the crossing was actually seen. `seq` is that frame's camera sequence number.
//!
//! An `Instant` has no meaning outside the process (it counts from an arbitrary
//! point), so [`WallClock`] converts it: it remembers one moment as both an `Instant`
//! and a `SystemTime`, and maps any other `Instant` by its distance from that moment.
//! Worked example: the anchor is `Instant` A = Unix 1 000 000 ms; a frame that arrived
//! 40 ms before A gets `t` = 999 960.
//!
//! Pure: no threads, no I/O. The sink in [`super`] does the writing.

use std::time::{Instant, SystemTime, UNIX_EPOCH};

use serde::Serialize;

use crate::detect::classes::CLASS_NAMES;
use crate::events::event::{Direction, Event, EventKind, ExitReason};

/// Maps `Instant`s to Unix milliseconds through one (Instant, SystemTime) pair taken at
/// startup.
#[derive(Clone, Copy, Debug)]
pub struct WallClock {
    /// The anchor moment, as an `Instant`…
    at: Instant,
    /// …and the same moment as Unix milliseconds.
    unix_ms: f64,
}

impl WallClock {
    /// Anchors the clock at the current moment.
    pub fn now() -> Self {
        Self::new(Instant::now(), SystemTime::now())
    }

    /// Anchors the clock at `at`, which is the wall-clock time `unix`.
    pub fn new(at: Instant, unix: SystemTime) -> Self {
        // A system clock set before 1970 is broken anyway; 0 keeps the times
        // relative to each other instead of failing.
        let since_epoch = unix.duration_since(UNIX_EPOCH).unwrap_or_default();
        Self {
            at,
            unix_ms: since_epoch.as_secs_f64() * 1000.0,
        }
    }

    /// The Unix time of `t` in whole milliseconds. `t` may be before or after the anchor.
    pub fn unix_ms(&self, t: Instant) -> i64 {
        (self.unix_ms + signed_ms(self.at, t)).round() as i64
    }
}

/// `to − from` in milliseconds, negative when `to` is earlier. `Instant` subtraction
/// panics when it would go negative, so try both directions with
/// `checked_duration_since`.
fn signed_ms(from: Instant, to: Instant) -> f64 {
    if let Some(ahead) = to.checked_duration_since(from) {
        ahead.as_secs_f64() * 1000.0
    } else if let Some(behind) = from.checked_duration_since(to) {
        -behind.as_secs_f64() * 1000.0
    } else {
        0.0
    }
}

/// The shape of one line. serde writes the fields in declaration order, so this order
/// *is* the schema's key order. The optional fields are left out entirely when `None`.
#[derive(Serialize)]
struct EventJson<'a> {
    #[serde(rename = "type")]
    kind: &'static str,
    t: i64,
    seq: u64,
    rule: u32,
    rule_name: &'a str,
    track: u32,
    class: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    dir: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    dwell_s: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    reason: Option<&'static str>,
}

/// One event as one JSON object, without the trailing newline.
pub fn event_line(e: &Event, clock: &WallClock) -> String {
    let mut json = EventJson {
        kind: "",
        t: clock.unix_ms(e.at),
        seq: e.seq,
        rule: e.rule,
        rule_name: &e.rule_name,
        track: e.track,
        class: class_name(e.class),
        dir: None,
        dwell_s: None,
        reason: None,
    };
    match e.kind {
        EventKind::LineCross { dir } => {
            json.kind = "line_cross";
            json.dir = Some(direction_name(dir));
        }
        EventKind::ZoneEnter => json.kind = "zone_enter",
        EventKind::ZoneDwell { dwell } => {
            json.kind = "zone_dwell";
            json.dwell_s = Some(tenths_of_a_second(dwell.as_secs_f64()));
        }
        EventKind::ZoneExit { dwell, reason } => {
            json.kind = "zone_exit";
            json.dwell_s = Some(tenths_of_a_second(dwell.as_secs_f64()));
            json.reason = Some(reason_name(reason));
        }
    }
    // Strings, integers and finite floats always serialise; only a map with non-string
    // keys or a failing custom `Serialize` could error, and there is neither here.
    serde_json::to_string(&json).expect("an event always serialises")
}

/// The COCO name of class `id`. The detector only produces ids below 80, but `get`
/// keeps a bad id from ever panicking the sink.
fn class_name(id: u8) -> &'static str {
    CLASS_NAMES
        .get(usize::from(id))
        .copied()
        .unwrap_or("unknown")
}

fn direction_name(dir: Direction) -> &'static str {
    match dir {
        Direction::AToB => "a_to_b",
        Direction::BToA => "b_to_a",
    }
}

fn reason_name(reason: ExitReason) -> &'static str {
    match reason {
        ExitReason::Left => "left",
        ExitReason::Lost => "lost",
    }
}

/// Rounds seconds to 0.1 s. People read these numbers, and millisecond precision on a
/// state that is debounced by 0.3 s would only be noise.
fn tenths_of_a_second(secs: f64) -> f64 {
    (secs * 10.0).round() / 10.0
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    /// A clock whose anchor `t0` is Unix 1 000 000 ms. `t0` is 10 s in the future so
    /// that `t0 − 40 ms` is a valid `Instant`.
    fn clock() -> (WallClock, Instant) {
        let t0 = Instant::now() + Duration::from_secs(10);
        let unix = UNIX_EPOCH + Duration::from_secs(1_000);
        (WallClock::new(t0, unix), t0)
    }

    fn event(kind: EventKind, rule: u32, name: &str, at: Instant) -> Event {
        Event {
            kind,
            rule,
            rule_name: name.to_string(),
            track: 7,
            class: 0,
            seq: 4512,
            at,
        }
    }

    #[test]
    fn emit_wall_clock_maps_past_and_future() {
        let (clock, t0) = clock();
        assert_eq!(clock.unix_ms(t0), 1_000_000);
        assert_eq!(clock.unix_ms(t0 + Duration::from_millis(250)), 1_000_250);
        assert_eq!(clock.unix_ms(t0 - Duration::from_millis(40)), 999_960);
    }

    #[test]
    fn emit_line_cross_json() {
        let (clock, t0) = clock();
        let kind = EventKind::LineCross {
            dir: Direction::AToB,
        };
        assert_eq!(
            event_line(&event(kind, 1, "door", t0), &clock),
            r#"{"type":"line_cross","t":1000000,"seq":4512,"rule":1,"rule_name":"door","track":7,"class":"person","dir":"a_to_b"}"#
        );
    }

    #[test]
    fn emit_zone_events_json() {
        let (clock, t0) = clock();
        let enter = event(EventKind::ZoneEnter, 2, "desk", t0);
        assert_eq!(
            event_line(&enter, &clock),
            r#"{"type":"zone_enter","t":1000000,"seq":4512,"rule":2,"rule_name":"desk","track":7,"class":"person"}"#
        );

        let dwell = EventKind::ZoneDwell {
            dwell: Duration::from_secs(10),
        };
        assert_eq!(
            event_line(&event(dwell, 2, "desk", t0), &clock),
            r#"{"type":"zone_dwell","t":1000000,"seq":4512,"rule":2,"rule_name":"desk","track":7,"class":"person","dwell_s":10.0}"#
        );

        let exit = EventKind::ZoneExit {
            dwell: Duration::from_millis(42_349),
            reason: ExitReason::Lost,
        };
        assert_eq!(
            event_line(&event(exit, 2, "desk", t0), &clock),
            r#"{"type":"zone_exit","t":1000000,"seq":4512,"rule":2,"rule_name":"desk","track":7,"class":"person","dwell_s":42.3,"reason":"lost"}"#
        );
    }

    #[test]
    fn emit_rule_name_is_escaped() {
        let (clock, t0) = clock();
        let name = "front \"door\"\nleft";
        let line = event_line(&event(EventKind::ZoneEnter, 3, name, t0), &clock);
        assert!(!line.contains('\n'), "the line must stay one line: {line}");
        let parsed: serde_json::Value = serde_json::from_str(&line).unwrap();
        assert_eq!(parsed["rule_name"], name);
    }

    #[test]
    fn emit_unknown_class_does_not_panic() {
        let (clock, t0) = clock();
        let mut e = event(EventKind::ZoneEnter, 1, "desk", t0);
        e.class = 200;
        assert!(event_line(&e, &clock).contains(r#""class":"unknown""#));
    }
}
