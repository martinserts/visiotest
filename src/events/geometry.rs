//! The 2D geometry behind line and zone rules, written by hand.
//!
//! Everything here works in **screen coordinates**: x grows to the right and y grows
//! *down*, as in the image. That flips some of the left/right intuitions from maths
//! class, so the side convention below is defined once and every other piece of code
//! asks [`signed_distance`] instead of working the sign out again.
//!
//! **Which side of a line?** Take the line from `a` to `b` and a point `p`. The 2D
//! cross product of the two vectors `b − a` and `p − a`,
//!
//! ```text
//! cross = (b.x − a.x)·(p.y − a.y) − (b.y − a.y)·(p.x − a.x)
//! ```
//!
//! is positive on one side of the line, negative on the other and zero on it. Its size
//! is `|b − a|` times the distance of `p` from the line, so dividing by `|b − a|` gives
//! the signed distance in pixels. We call the positive side **A**: A is on your right
//! when you stand on `a` and face `b`, in screen coordinates (y down). A line drawn
//! top → bottom therefore has A on the screen-left. Worked example:
//!
//! ```text
//!          a (960, 0)
//!          │
//!   p ●    │         cross = (960 − 960)·(500 − 0) − (1080 − 0)·(900 − 960)
//! (900,500)│               = 0·500 − 1080·(−60) = 64 800
//!   side A │ side B  |b − a| = 1080, so p is 64 800 / 1080 = +60 px from the line,
//!          │         on side A.
//!          b (960, 1080)
//! ```
//!
//! **Do two segments cross?** A crossing has to go *through* the drawn segment, not past
//! its end, so the infinite lines aren't enough. Segments `p0–p1` and `q0–q1` cross
//! exactly when `q0` and `q1` lie on opposite sides of the line through `p0p1` **and**
//! `p0` and `p1` lie on opposite sides of the line through `q0q1`: four cross-product
//! signs (the *orientation test*). When one of those signs is zero, that end point lies
//! on the other segment's line, and it touches the segment only if it also lies within
//! the segment's bounding box. Touching counts as crossing, so a walker whose anchor
//! lands exactly on a line's end point doesn't slip through.
//!
//! **Is a point inside a polygon?** Ray casting: shoot a ray from the point to the right
//! and count how many polygon edges it crosses. Every crossing swaps inside and outside,
//! so an odd count means inside:
//!
//! ```text
//!   ┌──────────┐
//!   │  ●───────┼──────►   1 crossing: inside
//!   │      ┌───┘
//!   │      │  ●──────►    0 crossings: outside (in the notch)
//!   └──────┘
//! ```
//!
//! An edge counts when its two ends lie on different sides of the ray, tested with the
//! half-open rule `(yi > p.y) != (yj > p.y)`. A vertex exactly at the ray's height then
//! belongs to only one of its two edges, so it is counted once, and horizontal edges
//! (both ends on the same side) are never counted. No epsilon is needed.
//!
//! **Convex or not?** egui only fills *convex* polygons correctly, so the UI fills a
//! zone only when [`is_convex`] says so and draws just the outline otherwise.

use serde::{Deserialize, Serialize};

use super::rules::Anchor;
use crate::detect::BBox;

/// A point in frame pixels, or in normalised frame coordinates (0..=1), as the caller says.
///
/// In `rules.json` it is written as `{ "x": …, "y": … }`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Point {
    pub x: f32,
    pub y: f32,
}

impl Point {
    pub fn new(x: f32, y: f32) -> Self {
        Self { x, y }
    }
}

/// The 2D cross product of `b − a` and `p − a`: positive when `p` is on side A of the
/// line `a → b`, negative on side B, zero on the line (see the module doc).
fn cross(a: Point, b: Point, p: Point) -> f32 {
    (b.x - a.x) * (p.y - a.y) - (b.y - a.y) * (p.x - a.x)
}

/// Signed distance of `p` from the infinite line through `a` → `b`, in `p`'s units.
///
/// **Positive = side A**: A is on your right when you stand on `a` and face `b`, with
/// screen coordinates (y down). Example: a = (960, 0), b = (960, 1080) (drawn top to
/// bottom); p = (900, 500) is 60 px on side A (screen-left). 0 when `a == b`.
pub fn signed_distance(a: Point, b: Point, p: Point) -> f32 {
    let length = (b.x - a.x).hypot(b.y - a.y);
    if length == 0.0 {
        // A zero-length line has no sides.
        return 0.0;
    }
    cross(a, b, p) / length
}

/// Whether segments p0–p1 and q0–q1 cross or touch (orientation test, collinear overlap counts).
pub fn segments_intersect(p0: Point, p1: Point, q0: Point, q1: Point) -> bool {
    // Which side of the other segment's line each end point is on.
    let q0_side = cross(p0, p1, q0);
    let q1_side = cross(p0, p1, q1);
    let p0_side = cross(q0, q1, p0);
    let p1_side = cross(q0, q1, p1);

    if opposite_signs(q0_side, q1_side) && opposite_signs(p0_side, p1_side) {
        return true;
    }
    // A zero means that end point lies on the other segment's (infinite) line; it
    // touches the segment itself only if it lies within its bounding box.
    (q0_side == 0.0 && within_box(p0, p1, q0))
        || (q1_side == 0.0 && within_box(p0, p1, q1))
        || (p0_side == 0.0 && within_box(q0, q1, p0))
        || (p1_side == 0.0 && within_box(q0, q1, p1))
}

/// True when one value is strictly positive and the other strictly negative.
fn opposite_signs(u: f32, v: f32) -> bool {
    (u > 0.0 && v < 0.0) || (u < 0.0 && v > 0.0)
}

/// Whether `p` lies in the bounding box of the segment `a–b`. For a point already known
/// to be on the segment's line, that means it is on the segment.
fn within_box(a: Point, b: Point, p: Point) -> bool {
    a.x.min(b.x) <= p.x && p.x <= a.x.max(b.x) && a.y.min(b.y) <= p.y && p.y <= a.y.max(b.y)
}

/// Ray casting (crossing number). Points exactly on an edge may go either way; the
/// zone debounce makes that harmless.
pub fn point_in_polygon(p: Point, poly: &[Point]) -> bool {
    let mut inside = false;
    // Walk the edges (j → i), starting with the closing edge from the last point back
    // to the first.
    let mut j = poly.len().wrapping_sub(1);
    for (i, &vi) in poly.iter().enumerate() {
        let vj = poly[j];
        // Half-open rule: the edge counts only if its ends are on different sides of the
        // ray's height. That also guarantees vi.y != vj.y below, so the division is safe.
        if (vi.y > p.y) != (vj.y > p.y) {
            // Where the edge meets the horizontal line y = p.y.
            let edge_x = vj.x + (p.y - vj.y) * (vi.x - vj.x) / (vi.y - vj.y);
            if p.x < edge_x {
                inside = !inside;
            }
        }
        j = i;
    }
    inside
}

/// True for a convex polygon (≥ 3 points, all turns the same way; collinear points allowed).
///
/// Checked as "every edge's line has the whole polygon on one side", with that side the
/// same for every edge. That is a little stronger than "all turns the same way", which a
/// five-pointed star (whose edges cross) also passes, and egui couldn't fill that either.
/// Zones have at most 32 points, so the n² cross products are cheap.
pub fn is_convex(poly: &[Point]) -> bool {
    if poly.len() < 3 {
        return false;
    }
    let mut seen_sign = 0.0_f32;
    for (i, &a) in poly.iter().enumerate() {
        let b = poly[(i + 1) % poly.len()];
        for &p in poly {
            let side = cross(a, b, p);
            if side == 0.0 {
                // On the edge's line: the edge's own ends, or a collinear point.
                continue;
            }
            if seen_sign == 0.0 {
                seen_sign = side.signum();
            } else if side.signum() != seen_sign {
                return false;
            }
        }
    }
    // All points on one line: no area to fill.
    seen_sign != 0.0
}

/// The point of a box that does the crossing.
pub fn anchor_point(b: BBox, anchor: Anchor) -> Point {
    let center_x = (b.x0 + b.x1) / 2.0;
    match anchor {
        Anchor::BottomCenter => Point::new(center_x, b.y1),
        Anchor::Center => Point::new(center_x, (b.y0 + b.y1) / 2.0),
    }
}

/// Normalised (0..=1) → frame pixels.
///
/// This scales x and y separately (an affine map), which keeps straight lines straight:
/// which side of a line a point is on and whether two segments cross come out the same
/// in either space. Distances don't, so a distance threshold (the crossing's dead band)
/// is measured in pixels.
pub fn to_pixels(p: Point, frame_w: f32, frame_h: f32) -> Point {
    Point::new(p.x * frame_w, p.y * frame_h)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pt(x: f32, y: f32) -> Point {
        Point::new(x, y)
    }

    #[test]
    fn events_signed_distance_sides() {
        let a = pt(960.0, 0.0);
        let b = pt(960.0, 1080.0);
        // The module doc's worked example: 60 px to the screen-left is side A.
        assert_eq!(signed_distance(a, b, pt(900.0, 500.0)), 60.0);
        // Its mirror image is on side B.
        assert_eq!(signed_distance(a, b, pt(1020.0, 500.0)), -60.0);
        assert_eq!(signed_distance(a, b, pt(960.0, 300.0)), 0.0);
        // A zero-length line has no sides.
        assert_eq!(signed_distance(a, a, pt(900.0, 500.0)), 0.0);
    }

    #[test]
    fn events_segments_intersect_cases() {
        // An X.
        assert!(segments_intersect(
            pt(0.0, 0.0),
            pt(10.0, 10.0),
            pt(0.0, 10.0),
            pt(10.0, 0.0)
        ));
        // Parallel, apart.
        assert!(!segments_intersect(
            pt(0.0, 0.0),
            pt(10.0, 0.0),
            pt(0.0, 5.0),
            pt(10.0, 5.0)
        ));
        // A T: one segment ends exactly on the other.
        assert!(segments_intersect(
            pt(0.0, 0.0),
            pt(10.0, 0.0),
            pt(5.0, 5.0),
            pt(5.0, 0.0)
        ));
        // Collinear and overlapping.
        assert!(segments_intersect(
            pt(0.0, 0.0),
            pt(10.0, 0.0),
            pt(5.0, 0.0),
            pt(15.0, 0.0)
        ));
        // Collinear but disjoint.
        assert!(!segments_intersect(
            pt(0.0, 0.0),
            pt(10.0, 0.0),
            pt(11.0, 0.0),
            pt(15.0, 0.0)
        ));
        // A move that passes beyond the end of a short line.
        assert!(!segments_intersect(
            pt(0.0, 20.0),
            pt(10.0, 20.0),
            pt(5.0, 0.0),
            pt(5.0, 10.0)
        ));
    }

    #[test]
    fn events_point_in_polygon_square_and_concave() {
        let square = [pt(0.0, 0.0), pt(1.0, 0.0), pt(1.0, 1.0), pt(0.0, 1.0)];
        assert!(point_in_polygon(pt(0.5, 0.5), &square));
        assert!(!point_in_polygon(pt(1.5, 0.5), &square));
        assert!(!point_in_polygon(pt(-0.5, 0.5), &square));
        assert!(!point_in_polygon(pt(0.5, 1.5), &square));

        // An L: a 2×2 square with its top-right quarter cut out.
        let l_shape = [
            pt(0.0, 0.0),
            pt(1.0, 0.0),
            pt(1.0, 1.0),
            pt(2.0, 1.0),
            pt(2.0, 2.0),
            pt(0.0, 2.0),
        ];
        assert!(!point_in_polygon(pt(1.5, 0.5), &l_shape), "in the notch");
        assert!(point_in_polygon(pt(0.5, 0.5), &l_shape), "upper arm");
        assert!(point_in_polygon(pt(1.5, 1.5), &l_shape), "lower arm");
    }

    #[test]
    fn events_is_convex() {
        let square = [pt(0.0, 0.0), pt(1.0, 0.0), pt(1.0, 1.0), pt(0.0, 1.0)];
        assert!(is_convex(&square));
        let l_shape = [
            pt(0.0, 0.0),
            pt(1.0, 0.0),
            pt(1.0, 1.0),
            pt(2.0, 1.0),
            pt(2.0, 2.0),
            pt(0.0, 2.0),
        ];
        assert!(!is_convex(&l_shape));
        assert!(is_convex(&[pt(0.0, 0.0), pt(4.0, 0.0), pt(2.0, 3.0)]));
        let square_with_midpoint = [
            pt(0.0, 0.0),
            pt(0.5, 0.0),
            pt(1.0, 0.0),
            pt(1.0, 1.0),
            pt(0.0, 1.0),
        ];
        assert!(is_convex(&square_with_midpoint));
        // Too few points, or no area.
        assert!(!is_convex(&[pt(0.0, 0.0), pt(1.0, 0.0)]));
        assert!(!is_convex(&[pt(0.0, 0.0), pt(1.0, 0.0), pt(2.0, 0.0)]));
        // A five-pointed star turns the same way at every corner, but its edges cross.
        let star = [
            pt(0.0, -1.0),
            pt(0.588, 0.809),
            pt(-0.951, -0.309),
            pt(0.951, -0.309),
            pt(-0.588, 0.809),
        ];
        assert!(!is_convex(&star));
    }

    #[test]
    fn events_anchor_points() {
        let b = BBox::new(100.0, 200.0, 300.0, 600.0);
        assert_eq!(anchor_point(b, Anchor::BottomCenter), pt(200.0, 600.0));
        assert_eq!(anchor_point(b, Anchor::Center), pt(200.0, 400.0));
    }

    #[test]
    fn events_to_pixels_scales_each_axis() {
        let (w, h) = (1920.0, 1080.0);
        assert_eq!(to_pixels(pt(0.0, 0.0), w, h), pt(0.0, 0.0));
        assert_eq!(to_pixels(pt(0.5, 0.5), w, h), pt(960.0, 540.0));
        assert_eq!(to_pixels(pt(1.0, 0.25), w, h), pt(1920.0, 270.0));
    }
}
