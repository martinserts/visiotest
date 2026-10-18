//! Tests for the overlay's pure helpers and the cursor rule.

use eframe::egui::CursorIcon;

use super::*;
use crate::detect::BBox;
use crate::events::event::Dweller;
use crate::events::geometry::signed_distance;
use crate::ui::detections::to_screen;
use crate::ui::rule_editor::Hit;

fn totals(a_to_b: u64, b_to_a: u64, visits: u64, inside: usize) -> RuleTotals {
    let since = Instant::now();
    RuleTotals {
        rule: 1,
        a_to_b,
        b_to_a,
        visits,
        inside: (1..=inside as u32)
            .map(|track| Dweller { track, since })
            .collect(),
        last_event: None,
    }
}

#[test]
fn rules_overlay_side_markers_match_engine() {
    // Top → bottom (side A is screen-left) and left → right (side A is below).
    let lines = [
        (pos2(500.0, 100.0), pos2(500.0, 600.0)),
        (pos2(100.0, 300.0), pos2(900.0, 300.0)),
    ];
    for (a, b) in lines {
        let m = side_markers(a, b);
        let side = |p: Pos2| signed_distance(to_point(a), to_point(b), to_point(p));
        assert!(side(m.a_label) > 0.0, "{a:?}→{b:?}: {m:?}");
        assert!(side(m.b_label) < 0.0, "{a:?}→{b:?}: {m:?}");
        assert!((side(m.a_label) - SIDE_OFFSET).abs() < 1e-3);
        // The arrow goes from side A to side B.
        assert!(side(m.arrow_from) > 0.0 && side(m.arrow_to) < 0.0, "{m:?}");
    }
    // Top → bottom: A is on the left of the line.
    let m = side_markers(pos2(500.0, 100.0), pos2(500.0, 600.0));
    assert!(m.a_label.x < 500.0 && m.b_label.x > 500.0, "{m:?}");
}

#[test]
fn rules_overlay_side_markers_zero_length() {
    let p = pos2(10.0, 20.0);
    let m = side_markers(p, p);
    assert_eq!((m.a_label, m.b_label), (p, p));
}

#[test]
fn rules_overlay_texts() {
    assert_eq!(line_count_text(&totals(5, 3, 0, 0)), "A→B 5 · B→A 3");
    assert_eq!(zone_text(&totals(0, 0, 7, 2)), "2 inside · 7 visits");
    assert_eq!(zone_text(&totals(0, 0, 1, 1)), "1 inside · 1 visit");
    assert_eq!(dwell_text(Duration::ZERO), "0:00");
    assert_eq!(dwell_text(Duration::from_millis(42_900)), "0:42");
    assert_eq!(dwell_text(Duration::from_secs(12 * 60 + 5)), "12:05");
    assert_eq!(
        dwell_text(Duration::from_secs(3600 + 2 * 60 + 9)),
        "1:02:09"
    );
}

#[test]
fn rules_overlay_flash_window() {
    let t = Instant::now();
    assert!(!flashing(None, t));
    assert!(flashing(Some(t), t));
    assert!(flashing(Some(t), t + Duration::from_millis(399)));
    assert!(!flashing(Some(t), t + FLASH));
    // An event stamped after `now` (never in practice) still flashes, no panic.
    assert!(flashing(Some(t + Duration::from_millis(5)), t));
}

#[test]
fn rules_overlay_viewport_matches_to_screen() {
    let video = Rect::from_min_size(pos2(100.0, 50.0), vec2(960.0, 540.0));
    let frame = vec2(1920.0, 1080.0);
    let norm = Point::new(0.25, 0.5);
    let px = (norm.x * frame.x, norm.y * frame.y);
    let want = to_screen(BBox::new(px.0, px.1, px.0, px.1), frame, video).min;
    let got = to_pos(viewport(video).to_screen(norm));
    assert!((got - want).length() < 1e-3, "{got:?} vs {want:?}");
}

#[test]
fn rules_overlay_cursor_follows_tool() {
    let hit = Some(Hit::Body { rule: 1 });
    assert_eq!(
        input::cursor_for(Tool::Line, false, None),
        CursorIcon::Crosshair
    );
    assert_eq!(
        input::cursor_for(Tool::Zone, true, hit),
        CursorIcon::Crosshair
    );
    assert_eq!(
        input::cursor_for(Tool::Select, false, hit),
        CursorIcon::Grab
    );
    assert_eq!(
        input::cursor_for(Tool::Select, true, hit),
        CursorIcon::Grabbing
    );
    assert_eq!(
        input::cursor_for(Tool::Select, false, None),
        CursorIcon::Default
    );
}
