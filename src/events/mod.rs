//! Events: turn tracks into "#7 crossed the door line A→B" and "#7 stayed at the desk 42 s".
//!
//! The tracker says where each object is *now*: "#7 is a person at (900, 500)". A camera
//! system wants something else: "#7 walked through the door, going in" and "#7 has been
//! at the desk for 42 s". This module turns the first into the second, with rules the
//! user draws on the video:
//!
//! - a **line** (a tripwire): every tracked object counts once when it crosses it, and
//!   the two directions, `A→B` and `B→A`, are counted separately;
//! - a **zone** (a polygon): objects entering it, staying longer than the zone's dwell
//!   time, and leaving it again.
//!
//! **It's all 2D.** A line or a zone is drawn on the *image*, and every test is plain 2D
//! geometry in frame pixels, with no 3D and no camera calibration. That works because a
//! flat floor maps to the image with straight lines kept straight: when the floor is
//! visible, a line drawn on the image of the floor *is* a line on the floor. So the
//! point of a box that does the crossing, the **anchor**, is by default its
//! **bottom-centre** (the "feet"): the feet crossing the image line means the person
//! crossed the floor line. With a desk view, where the frame cuts people off at the
//! chest and no floor is visible, use the box **centre** instead.
//!
//! ```text
//! tracks of one detector run (Tracked + Lost, at the frame's time)
//!      │
//!      ▼
//! engine ──► line rules: did the anchor's side flip, through the drawn segment?
//!        └─► zone rules: is the anchor inside the polygon? for how long?
//!      │
//!      ▼
//! events ("line_cross", "zone_enter", "zone_dwell", "zone_exit")
//!      │
//!      ├──► one JSON line each on stdout
//!      └──► running counts for the UI
//! ```
//!
//! - [`geometry`] is the maths: which side of a line a point is on, whether two
//!   segments cross, whether a point is inside a polygon, and the box anchor.
//! - [`rules`] holds the rules themselves (lines and zones, in normalised `0..=1` frame
//!   coordinates, each with its own class filter) and reads and writes `rules.json`.
//! - [`event`] defines the events and the running totals the engine hands out.
//! - [`line`] is the line-crossing state machine of one track: a confident side, a dead
//!   band around the line, and a move that must go through the drawn segment.
//! - [`zone`] is the zone state machine of one track: enter and exit debounced in time,
//!   one dwell event per visit, and the dwell clock running on while the track is hidden.
//! - [`engine`] runs both on every detector result: rules in pixels, one memory per rule
//!   and track, the events of the run and the running totals.

pub mod engine;
pub mod event;
pub mod geometry;
pub mod line;
pub mod rules;
#[cfg(test)]
mod sim_tests;
pub mod zone;
