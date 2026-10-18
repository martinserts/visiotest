//! The line-crossing state machine: when has a track crossed a line, and which way?
//!
//! **The naive rule fails.** "The anchor was on side A last run and is on side B now"
//! sounds right, but the anchor of a box never stands still: the detector's box jitters
//! by a few pixels from frame to frame. Someone standing *on* the line would flip sides
//! every frame and be counted dozens of times. And someone walking past the *end* of a
//! short line also changes side (the side test is about the infinite line), without
//! ever going through the segment that was drawn.
//!
//! **The rule used here** fixes both with two extra conditions:
//!
//! ```text
//!            A │ B
//!   ●──►──●────┼───►●        a crossing: the confident side flipped (A → B) AND
//!   last       │    now      the move last-confident → now went through the segment
//!   confident  │
//!        ┊┊┊┊┊┊│┊┊┊┊┊┊        dead band: within the margin of the line, the side is
//!              │             "not sure", so jitter on the line never flips it
//! ```
//!
//! 1. **A dead band (hysteresis).** Within `margin` pixels of the line the side is
//!    "not sure" and nothing changes. Each track remembers its last *confident* side,
//!    and the point where it was last seen there. Jitter around the line stays inside
//!    the band, so it never flips the side.
//! 2. **The movement segment must cross the drawn segment.** When the confident side
//!    flips, the move from the last confident point to the new one is a segment of its
//!    own. Only if it intersects the line's segment
//!    ([`super::geometry::segments_intersect`]) is it a crossing. Walking around the end
//!    of the line flips the side too, but the move misses the segment, so it isn't
//!    counted (the side is still updated, so a later real crossing back counts).
//!
//! Using the whole move, not just the last step, also catches big jumps: with the
//! detector paced to 500 ms, a walker at 1000 px/s moves 500 px between runs, from well
//! on side A to well on side B, and the move still goes through the line.
//!
//! **Worked example.** A vertical line at x 960 (drawn top → bottom, so A is the
//! screen-left), margin 10 px. The anchor is seen at x 900 (A, confident: remember it),
//! 955 (in the band: ignore), 965 (in the band: ignore), 1020 (B, confident). The side
//! flipped A → B and the move 900 → 1020 goes through the line: one `AToB`.
//!
//! **A track that first appears on a side has not crossed.** The first confident
//! sighting only sets the side. That is also why the engine never feeds a Lost track's
//! guessed position in here: its side freezes while it is hidden, and the crossing is
//! counted when it is found again on the other side.

use super::event::Direction;
use super::geometry::{Point, segments_intersect, signed_distance};
use crate::track::TrackId;

/// Which side of a line a point is on: A is the positive side of [`signed_distance`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Side {
    A,
    B,
}

/// A line rule in frame pixels.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PixelLine {
    pub a: Point,
    pub b: Point,
}

/// What a line rule remembers about one track.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LineMemory {
    pub track: TrackId,
    /// The last confident side; `None` until the track is first seen outside the band.
    side: Option<Side>,
    /// Where the anchor was the last time it was confidently on `side`.
    last_confident: Point,
    /// Whether the current detector run had a view of this track (the engine's
    /// bookkeeping: memories not seen in a run belong to deleted tracks).
    pub seen: bool,
}

impl LineMemory {
    /// A fresh memory: no side known yet.
    pub fn new(track: TrackId) -> Self {
        Self {
            track,
            side: None,
            last_confident: Point::default(),
            seen: false,
        }
    }

    /// One Tracked sighting of the anchor at `p` (frame pixels). Returns the direction
    /// when this sighting completes a crossing.
    ///
    /// `margin` is the dead band's half-width in pixels: within it of the line, the
    /// side is "not sure" and the memory is left as it is.
    pub fn observe(&mut self, line: &PixelLine, p: Point, margin: f32) -> Option<Direction> {
        let d = signed_distance(line.a, line.b, p);
        if d.abs() < margin {
            return None;
        }
        let now = if d > 0.0 { Side::A } else { Side::B };
        let Some(before) = self.side else {
            // The first confident sighting: the track appeared here, it didn't cross.
            self.remember(now, p);
            return None;
        };
        if now == before {
            self.remember(now, p);
            return None;
        }
        // The confident side flipped. It is a crossing only if the move went through
        // the drawn segment, not around one of its ends.
        let crossed = segments_intersect(self.last_confident, p, line.a, line.b);
        self.remember(now, p);
        if !crossed {
            return None;
        }
        Some(match before {
            Side::A => Direction::AToB,
            Side::B => Direction::BToA,
        })
    }

    fn remember(&mut self, side: Side, p: Point) {
        self.side = Some(side);
        self.last_confident = p;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A 10 px band: the engine's dead band for a box 100 px on its smaller side.
    const MARGIN: f32 = 10.0;

    /// A vertical line at x 960 drawn top → bottom, so A is the screen-left.
    fn vertical() -> PixelLine {
        PixelLine {
            a: Point::new(960.0, 0.0),
            b: Point::new(960.0, 1080.0),
        }
    }

    /// Feeds the anchor positions `(x, y)` in order and collects the crossings.
    fn walk(memory: &mut LineMemory, line: &PixelLine, path: &[(f32, f32)]) -> Vec<Direction> {
        path.iter()
            .filter_map(|&(x, y)| memory.observe(line, Point::new(x, y), MARGIN))
            .collect()
    }

    #[test]
    fn events_line_simple_crossing() {
        let line = vertical();
        let mut m = LineMemory::new(1);
        let there: Vec<(f32, f32)> = (0..=12).map(|i| (900.0 + 10.0 * i as f32, 500.0)).collect();
        assert_eq!(walk(&mut m, &line, &there), [Direction::AToB]);
        let back: Vec<(f32, f32)> = there.iter().rev().copied().collect();
        assert_eq!(walk(&mut m, &line, &back), [Direction::BToA]);
    }

    #[test]
    fn events_line_jitter_in_dead_band() {
        let line = vertical();
        let mut m = LineMemory::new(1);
        // Confidently on A first, so a flip would be possible.
        assert_eq!(walk(&mut m, &line, &[(900.0, 500.0)]), []);
        let jitter: Vec<(f32, f32)> = (0..200)
            .map(|i| (if i % 2 == 0 { 955.0 } else { 965.0 }, 500.0))
            .collect();
        assert_eq!(walk(&mut m, &line, &jitter), []);
    }

    #[test]
    fn events_line_first_sighting_on_b_no_event() {
        let line = vertical();
        let mut m = LineMemory::new(1);
        assert_eq!(walk(&mut m, &line, &[(1020.0, 500.0), (1100.0, 500.0)]), []);
    }

    #[test]
    fn events_line_past_the_end() {
        // A short line from y 400 to y 600.
        let line = PixelLine {
            a: Point::new(960.0, 400.0),
            b: Point::new(960.0, 600.0),
        };
        let mut m = LineMemory::new(1);
        // From A to B below the line's end: the side flips, but nothing crossed.
        assert_eq!(walk(&mut m, &line, &[(900.0, 800.0), (1020.0, 800.0)]), []);
        // The side did update: going back through the segment counts as B → A.
        assert_eq!(
            walk(&mut m, &line, &[(1020.0, 500.0), (900.0, 500.0)]),
            [Direction::BToA]
        );
    }

    #[test]
    fn events_line_big_jump() {
        // 500 ms pacing at ~1000 px/s: one run on A, the next far on B.
        let line = vertical();
        let mut m = LineMemory::new(1);
        assert_eq!(
            walk(&mut m, &line, &[(700.0, 500.0), (1220.0, 500.0)]),
            [Direction::AToB]
        );
    }
}
