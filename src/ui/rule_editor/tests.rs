//! Tests for the rule editor: every interaction, driven through `on_input` (pointer
//! inputs in screen points, as the UI sends them) and `on_key`.

use super::*;

/// Half of a 1920×1080 frame, placed like a video rect next to a panel.
const VIEW: Viewport = Viewport {
    origin: Point { x: 100.0, y: 50.0 },
    width: 960.0,
    height: 540.0,
};

fn pt(x: f32, y: f32) -> Point {
    Point::new(x, y)
}

fn press(x: f32, y: f32) -> EditorInput {
    EditorInput::Press(pt(x, y))
}

fn drag(x: f32, y: f32) -> EditorInput {
    EditorInput::Drag {
        pos: pt(x, y),
        snap: false,
    }
}

fn release() -> EditorInput {
    EditorInput::Release
}

/// Feeds `inputs` in order and returns the last outcome.
fn feed(editor: &mut Editor, rules: &mut Rules, inputs: &[EditorInput]) -> Outcome {
    let mut last = Outcome::default();
    for &input in inputs {
        last = editor.on_input(input, rules, &VIEW);
    }
    last
}

/// Feeds one key and returns its outcome.
fn key(editor: &mut Editor, rules: &mut Rules, k: EditorKey) -> Outcome {
    editor.on_key(k, rules)
}

fn assert_near(actual: Point, expected: Point) {
    let close = (actual.x - expected.x).abs() < 1e-3 && (actual.y - expected.y).abs() < 1e-3;
    assert!(close, "{actual:?} is not near {expected:?}");
}

/// The normalised points of a 0.2-wide triangle with its top-left at (`x`, `y`).
fn triangle(x: f32, y: f32) -> Vec<Point> {
    vec![pt(x, y), pt(x + 0.2, y), pt(x + 0.1, y + 0.2)]
}

#[test]
fn rule_editor_draws_a_line() {
    let (mut editor, mut rules) = (Editor::new(), Rules::default());
    editor.set_tool(Tool::Line);

    let out = feed(
        &mut editor,
        &mut rules,
        &[press(200.0, 100.0), drag(600.0, 100.0)],
    );
    assert!(!out.changed, "a draft is not a change");
    assert!(matches!(editor.draft(), Some(Draft::Line { .. })));

    let out = feed(&mut editor, &mut rules, &[release()]);
    assert!(out.changed);
    assert_eq!(rules.lines.len(), 1);
    let line = &rules.lines[0];
    // (200 − 100) / 960 = 0.104, (100 − 50) / 540 = 0.093, (600 − 100) / 960 = 0.521.
    assert_near(line.a, pt(0.104_17, 0.092_59));
    assert_near(line.b, pt(0.520_83, 0.092_59));
    assert_eq!(editor.selected, Some(line.id));
    assert_eq!(editor.tool, Tool::Select);
    assert_eq!(editor.draft(), None);
}

#[test]
fn rule_editor_short_line_discarded() {
    let (mut editor, mut rules) = (Editor::new(), Rules::default());
    editor.set_tool(Tool::Line);
    let inputs = [press(200.0, 100.0), drag(205.0, 100.0), release()];
    let out = feed(&mut editor, &mut rules, &inputs);
    assert!(!out.changed);
    assert!(rules.is_empty());
    assert_eq!(editor.draft(), None);
    assert_eq!(editor.tool, Tool::Line, "ready for a real drag");
}

#[test]
fn rule_editor_shift_snaps() {
    let (mut editor, mut rules) = (Editor::new(), Rules::default());
    editor.set_tool(Tool::Line);
    let snapped = EditorInput::Drag {
        pos: pt(600.0, 130.0),
        snap: true,
    };
    feed(
        &mut editor,
        &mut rules,
        &[press(200.0, 100.0), snapped, release()],
    );
    let line = &rules.lines[0];
    assert_eq!(line.b.y, line.a.y, "|dx| > |dy|, so the line is horizontal");
    assert_near(line.b, pt(0.520_83, 0.092_59));

    // Mostly downwards: vertical.
    editor.set_tool(Tool::Line);
    let down = EditorInput::Drag {
        pos: pt(230.0, 400.0),
        snap: true,
    };
    feed(
        &mut editor,
        &mut rules,
        &[press(200.0, 100.0), down, release()],
    );
    let line = &rules.lines[1];
    assert_eq!(line.b.x, line.a.x);
}

#[test]
fn rule_editor_zone_by_clicks_and_close() {
    let (mut editor, mut rules) = (Editor::new(), Rules::default());
    editor.set_tool(Tool::Zone);

    feed(
        &mut editor,
        &mut rules,
        &[press(200.0, 100.0), press(500.0, 100.0)],
    );
    // Enter with two corners: not a zone yet.
    let out = key(&mut editor, &mut rules, EditorKey::Finish);
    assert!(!out.changed);
    assert!(rules.is_empty());

    // The rubber-band edge follows the pointer.
    feed(
        &mut editor,
        &mut rules,
        &[EditorInput::Hover(pt(340.0, 320.0))],
    );
    let Some(Draft::Zone { points, hover }) = editor.draft() else {
        panic!("expected a zone draft");
    };
    assert_eq!(points.len(), 2);
    assert_near(hover.unwrap(), VIEW.to_norm(pt(340.0, 320.0)));

    // A third corner, then a click within the handle radius of the first closes it.
    feed(&mut editor, &mut rules, &[press(300.0, 400.0)]);
    let out = feed(&mut editor, &mut rules, &[press(203.0, 102.0)]);
    assert!(out.changed);
    assert_eq!(rules.zones.len(), 1);
    assert_eq!(rules.zones[0].points.len(), 3);
    assert_eq!(editor.selected, Some(rules.zones[0].id));
    assert_eq!(editor.tool, Tool::Select);
    assert_eq!(editor.draft(), None);

    // Enter closes a zone with three corners too.
    editor.set_tool(Tool::Zone);
    let clicks = [
        press(600.0, 100.0),
        press(900.0, 100.0),
        press(700.0, 400.0),
    ];
    feed(&mut editor, &mut rules, &clicks);
    let out = key(&mut editor, &mut rules, EditorKey::Finish);
    assert!(out.changed);
    assert_eq!(rules.zones.len(), 2);
}

#[test]
fn rule_editor_zone_autocloses_at_max() {
    let (mut editor, mut rules) = (Editor::new(), Rules::default());
    editor.set_tool(Tool::Zone);
    // A zig-zag: no click lands near the first corner.
    for i in 0..MAX_ZONE_POINTS {
        let (x, y) = (150.0 + i as f32 * 25.0, 100.0 + (i % 2) as f32 * 100.0);
        let out = feed(&mut editor, &mut rules, &[press(x, y)]);
        assert_eq!(out.changed, i + 1 == MAX_ZONE_POINTS, "click {i}");
    }
    assert_eq!(rules.zones.len(), 1);
    assert_eq!(rules.zones[0].points.len(), MAX_ZONE_POINTS);
    assert_eq!(editor.tool, Tool::Select);
}

#[test]
fn rule_editor_moves_handle_and_body() {
    let (mut editor, mut rules) = (Editor::new(), Rules::default());
    let line = rules.add_line(pt(0.1, 0.1), pt(0.3, 0.1));
    let zone = rules.add_zone(triangle(0.4, 0.4));

    // Drag the line's end `b` 96 points right (0.1 of the width).
    let b = VIEW.to_screen(pt(0.3, 0.1));
    let out = feed(
        &mut editor,
        &mut rules,
        &[press(b.x, b.y), drag(b.x + 96.0, b.y)],
    );
    assert!(out.changed);
    assert!(editor.is_dragging());
    assert!(!feed(&mut editor, &mut rules, &[release()]).changed);
    assert!(!editor.is_dragging());
    assert_near(rules.lines[0].b, pt(0.4, 0.1));
    assert_near(rules.lines[0].a, pt(0.1, 0.1));
    assert_eq!(editor.selected, Some(line));

    // Drag the zone's body by (+0.1, +0.1): every corner moves by the same amount.
    let inside = VIEW.to_screen(pt(0.5, 0.45));
    let to = pt(inside.x + 96.0, inside.y + 54.0);
    let out = feed(
        &mut editor,
        &mut rules,
        &[press(inside.x, inside.y), drag(to.x, to.y)],
    );
    assert!(out.changed);
    assert_eq!(editor.selected, Some(zone));
    for (moved, start) in rules.zones[0].points.iter().zip(triangle(0.4, 0.4)) {
        assert_near(*moved, pt(start.x + 0.1, start.y + 0.1));
    }

    // Keep dragging far past the right edge: the shape stops at the edge, intact.
    feed(
        &mut editor,
        &mut rules,
        &[drag(to.x + 2000.0, to.y), release()],
    );
    let points = &rules.zones[0].points;
    let max_x = points.iter().map(|p| p.x).fold(f32::MIN, f32::max);
    let min_x = points.iter().map(|p| p.x).fold(f32::MAX, f32::min);
    assert_near(pt(max_x, 0.0), pt(1.0, 0.0));
    assert_near(pt(max_x - min_x, 0.0), pt(0.2, 0.0));
    // y is where the earlier drag left it.
    assert_near(points[0], pt(0.8, 0.5));
}

#[test]
fn rule_editor_line_handle_refuses_collapse() {
    let (mut editor, mut rules) = (Editor::new(), Rules::default());
    rules.add_line(pt(0.1, 0.1), pt(0.3, 0.1));
    let (a, b) = (VIEW.to_screen(pt(0.1, 0.1)), VIEW.to_screen(pt(0.3, 0.1)));
    let out = feed(&mut editor, &mut rules, &[press(b.x, b.y), drag(a.x, a.y)]);
    assert!(!out.changed);
    assert_eq!(rules.lines[0].b, pt(0.3, 0.1), "keeps the last position");
}

#[test]
fn rule_editor_handle_wins_over_body() {
    let mut rules = Rules::default();
    let zone = rules.add_zone(vec![pt(0.1, 0.1), pt(0.9, 0.1), pt(0.5, 0.9)]);
    // A line whose end `a` lies inside the zone.
    let line = rules.add_line(pt(0.5, 0.3), pt(0.95, 0.95));

    let on_a = VIEW.to_screen(pt(0.5, 0.3));
    let hit = Editor::hit_test(&rules, pt(on_a.x + 3.0, on_a.y), &VIEW);
    assert_eq!(
        hit,
        Some(Hit::Handle {
            rule: line,
            index: 0
        })
    );

    // The zone's own corner is a handle, not the body it sits on.
    let corner = VIEW.to_screen(pt(0.9, 0.1));
    let hit = Editor::hit_test(&rules, corner, &VIEW);
    assert_eq!(
        hit,
        Some(Hit::Handle {
            rule: zone,
            index: 1
        })
    );

    // Away from any handle: the body.
    let inside = VIEW.to_screen(pt(0.3, 0.2));
    assert_eq!(
        Editor::hit_test(&rules, inside, &VIEW),
        Some(Hit::Body { rule: zone })
    );
    // Beyond the line's end: nothing, although it is on the line's infinite extension.
    let past_end = VIEW.to_screen(pt(0.05, -0.1));
    assert_eq!(Editor::hit_test(&rules, past_end, &VIEW), None);
}

#[test]
fn rule_editor_newest_rule_wins() {
    let mut rules = Rules::default();
    rules.add_zone(triangle(0.4, 0.4));
    let newer_zone = rules.add_zone(triangle(0.45, 0.4));
    let inside_both = VIEW.to_screen(pt(0.55, 0.45));
    assert_eq!(
        Editor::hit_test(&rules, inside_both, &VIEW),
        Some(Hit::Body { rule: newer_zone })
    );

    // Two lines crossing at (0.5, 0.2): the newer one is on top.
    rules.add_line(pt(0.3, 0.2), pt(0.7, 0.2));
    let newer_line = rules.add_line(pt(0.5, 0.05), pt(0.5, 0.35));
    let crossing = VIEW.to_screen(pt(0.5, 0.2));
    assert_eq!(
        Editor::hit_test(&rules, crossing, &VIEW),
        Some(Hit::Body { rule: newer_line })
    );
}

#[test]
fn rule_editor_delete_and_right_click_corner() {
    let (mut editor, mut rules) = (Editor::new(), Rules::default());
    let square = vec![pt(0.1, 0.1), pt(0.3, 0.1), pt(0.3, 0.3), pt(0.1, 0.3)];
    let zone = rules.add_zone(square);
    rules.add_zone(triangle(0.6, 0.6));

    // Right-click a corner of the 4-corner zone: removed.
    let corner = VIEW.to_screen(pt(0.3, 0.3));
    let out = feed(
        &mut editor,
        &mut rules,
        &[EditorInput::SecondaryClick(corner)],
    );
    assert!(out.changed);
    assert_eq!(rules.zones[0].points.len(), 3);
    assert!(!rules.zones[0].points.contains(&pt(0.3, 0.3)));

    // The 3-corner one keeps its corners.
    let corner = VIEW.to_screen(pt(0.6, 0.6));
    let out = feed(
        &mut editor,
        &mut rules,
        &[EditorInput::SecondaryClick(corner)],
    );
    assert!(!out.changed);
    assert_eq!(rules.zones[1].points.len(), 3);

    // Delete with nothing selected does nothing; with a selection it removes the rule.
    assert!(!key(&mut editor, &mut rules, EditorKey::Delete).changed);
    let inside = VIEW.to_screen(pt(0.15, 0.15));
    feed(
        &mut editor,
        &mut rules,
        &[press(inside.x, inside.y), release()],
    );
    let out = key(&mut editor, &mut rules, EditorKey::Delete);
    assert!(out.changed);
    assert_eq!(rules.name(zone), None);
    assert_eq!(rules.zones.len(), 1);
    assert_eq!(editor.selected, None);
}

#[test]
fn rule_editor_escape_order() {
    let (mut editor, mut rules) = (Editor::new(), Rules::default());
    editor.toggle_editing();
    editor.set_tool(Tool::Zone);
    let clicks = [
        press(200.0, 100.0),
        press(500.0, 100.0),
        press(300.0, 400.0),
    ];
    feed(&mut editor, &mut rules, &clicks);
    let id = rules.add_line(pt(0.1, 0.8), pt(0.3, 0.8));
    editor.select(id);

    // 1. The half-drawn zone goes; the selection stays.
    let out = key(&mut editor, &mut rules, EditorKey::Cancel);
    assert_eq!(out, Outcome::default());
    assert_eq!(editor.draft(), None);
    assert_eq!(editor.selected, Some(id));
    assert!(rules.zones.is_empty());
    assert!(editor.is_editing());

    // 2. Then the selection.
    let out = key(&mut editor, &mut rules, EditorKey::Cancel);
    assert_eq!(out, Outcome::default());
    assert_eq!(editor.selected, None);
    assert!(editor.is_editing());

    // 3. Then edit mode itself.
    let out = key(&mut editor, &mut rules, EditorKey::Cancel);
    assert!(!editor.is_editing());
    assert!(!out.changed);
}

#[test]
fn rule_editor_set_tool_drops_draft_keeps_selection() {
    let (mut editor, mut rules) = (Editor::new(), Rules::default());
    editor.toggle_editing();
    editor.select(rules.add_line(pt(0.1, 0.8), pt(0.3, 0.8)));
    editor.set_tool(Tool::Line);
    feed(
        &mut editor,
        &mut rules,
        &[press(200.0, 100.0), drag(600.0, 100.0)],
    );
    editor.set_tool(Tool::Zone);
    assert_eq!(editor.draft(), None);
    assert!(editor.selected.is_some());
    // The old drag is gone too: a release adds nothing.
    assert!(!feed(&mut editor, &mut rules, &[release()]).changed);
    assert_eq!(rules.len(), 1);
}

#[test]
fn rule_editor_toggle_editing_resets() {
    let (mut editor, mut rules) = (Editor::new(), Rules::default());
    let id = rules.add_line(pt(0.1, 0.8), pt(0.3, 0.8));
    editor.toggle_editing();
    assert!(editor.is_editing());
    editor.set_tool(Tool::Zone);
    editor.select(id);
    editor.toggle_editing();
    assert!(!editor.is_editing());
    assert_eq!(editor.tool(), Tool::Select);
    assert_eq!(editor.selected(), None);
}

#[test]
fn rule_editor_select_only_in_edit_mode() {
    let (mut editor, mut rules) = (Editor::new(), Rules::default());
    let id = rules.add_line(pt(0.1, 0.8), pt(0.3, 0.8));
    // A click on a card row outside edit mode selects nothing.
    editor.select(id);
    assert_eq!(editor.selected(), None);
    editor.toggle_editing();
    editor.select(id);
    assert_eq!(editor.selected(), Some(id));
    // Leaving edit mode clears it again.
    editor.toggle_editing();
    assert_eq!(editor.selected(), None);
}

#[test]
fn rule_editor_remove_forgets_selection_and_drag() {
    let (mut editor, mut rules) = (Editor::new(), Rules::default());
    let id = rules.add_zone(triangle(0.4, 0.4));

    // Grab the zone's body, then the card's 🗑 removes it mid-drag.
    let inside = VIEW.to_screen(pt(0.5, 0.45));
    feed(&mut editor, &mut rules, &[press(inside.x, inside.y)]);
    assert_eq!(editor.selected, Some(id));
    assert!(editor.remove(id, &mut rules));
    assert_eq!(editor.selected, None);

    feed(&mut editor, &mut rules, &[drag(inside.x + 50.0, inside.y)]);
    let out = key(&mut editor, &mut rules, EditorKey::Delete);
    assert!(!out.changed);
    assert_eq!(editor.selected, None);
    assert!(rules.is_empty());
}

#[test]
fn rule_editor_press_on_nothing_deselects() {
    let (mut editor, mut rules) = (Editor::new(), Rules::default());
    let id = rules.add_zone(triangle(0.4, 0.4));
    let inside = VIEW.to_screen(pt(0.5, 0.45));
    feed(&mut editor, &mut rules, &[press(inside.x, inside.y)]);
    assert_eq!(editor.selected, Some(id));
    feed(&mut editor, &mut rules, &[release()]);
    feed(&mut editor, &mut rules, &[press(120.0, 500.0)]);
    assert_eq!(editor.selected, None);
}

#[test]
fn rule_editor_viewport_round_trip_and_clamp() {
    let p = pt(0.25, 0.75);
    let screen = VIEW.to_screen(p);
    assert_near(screen, pt(340.0, 455.0));
    assert_near(VIEW.to_norm(screen), p);

    // Past the video's edges: pinned to them.
    assert_eq!(VIEW.to_norm(pt(0.0, 9999.0)), pt(0.0, 1.0));
    assert_eq!(VIEW.to_norm(pt(5000.0, -20.0)), pt(1.0, 0.0));

    // An empty rect (no frame yet) doesn't divide by zero.
    let empty = Viewport {
        origin: pt(0.0, 0.0),
        width: 0.0,
        height: 0.0,
    };
    assert_eq!(empty.to_norm(pt(10.0, 10.0)), pt(0.0, 0.0));
}
