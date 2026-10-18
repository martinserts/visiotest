//! Tracking: link detections across frames into objects with stable IDs.
//!
//! The detector sees every frame on its own: it can say "there is a person here", but
//! not "this is the *same* person as last frame". The tracker adds that memory. It keeps
//! a list of *tracks* (objects it has seen before), and for each new set of detections
//! it decides which detection belongs to which track:
//!
//! ```text
//! detections of one frame
//!      │
//!      ▼
//! predict every track forward (kalman.rs)  ──►  cost table: tracks × detections (association.rs)
//!                                                     │
//!                                                     ▼
//!                                     best one-to-one pairs (assignment.rs = Hungarian)
//!                                                     │
//!                                                     ▼
//!                  update matched tracks, start / lose / delete the rest (tracker.rs)
//! ```
//!
//! - [`kalman`] moves each track's box forward to the new frame's time, so we compare
//!   the detections with where each object *should be now*, not where it was.
//! - [`association`] scores every track × detection pair by how much their boxes
//!   overlap (IoU), fills a cost table and keeps only pairs that overlap enough.
//! - [`assignment`] picks the pairs with the smallest total cost, each track and each
//!   detection used at most once.
//! - [`tracker`] runs the rounds below, blends each matched detection into its track's
//!   filter, starts new tracks and forgets tracks gone too long.
//!
//! **The whole ByteTrack loop**, once per detector run:
//!
//! 1. **Predict** every track to the new frame's time (Kalman).
//! 2. **Split** the detections: **high** (≥ the confidence slider) and **low**
//!    (≥ 0.10).
//! 3. **Round 1:** confirmed tracks (Tracked + Lost) vs the high detections.
//! 4. **Round 2:** still-unmatched *Tracked* tracks vs the **low** detections. This is
//!    ByteTrack's idea: a half-hidden person often still gets a 0.2 box, and most
//!    trackers would throw it away. Tracked tracks still unmatched now become Lost.
//! 5. **Round 3:** tentative tracks vs the high detections left over. A second hit
//!    confirms a track and gives it its id; a miss deletes it.
//! 6. **Start** a tentative track for each high detection still left. **Delete** Lost
//!    tracks after 1 s *and* 3 missed runs.
//!
//! Two test-only modules measure the tracker:
//!
//! - [`eval`] scores tracks against ground truth: ID switches, misses, false positives,
//!   MOTA (the CLEAR-MOT basics).
//! - [`sim`] plays deterministic synthetic scenes (walkers, poles, walls, crossings)
//!   through the tracker and the detect-interval pacing, and prints the ID-switch
//!   benchmark: `cargo test --release sim_report -- --ignored --nocapture`.

pub mod assignment;
pub mod association;
#[cfg(test)] // benchmark-only until a real, labelled video exists
pub mod eval;
pub mod kalman;
#[cfg(test)]
pub mod sim;
pub mod tracker;
mod view;

/// A track's identity, shown on screen as `#7`.
///
/// Handed out as 1, 2, 3, … in the order tracks are confirmed, and never reused, so an
/// ID that changes for the same object is a real ID switch, not a recycled number.
pub type TrackId = u32;
