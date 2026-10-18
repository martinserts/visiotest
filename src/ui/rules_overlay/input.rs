//! The edit mode's input: egui's pointer events over the video, turned into the rule
//! editor's [`EditorInput`]s, and the pointer's shape.

use eframe::egui::{self, CursorIcon, PointerButton, Rect, Sense};

use super::{to_point, viewport};
use crate::events::rules::Rules;
use crate::ui::rule_editor::{Editor, EditorInput, Hit, Outcome, Tool, Viewport};

/// The pointer's shape: a crosshair while drawing, an open hand over something the
/// Select tool can grab, a closed hand while dragging it.
pub fn cursor_for(tool: Tool, dragging: bool, hover: Option<Hit>) -> CursorIcon {
    match tool {
        Tool::Line | Tool::Zone => CursorIcon::Crosshair,
        Tool::Select if dragging => CursorIcon::Grabbing,
        Tool::Select if hover.is_some() => CursorIcon::Grab,
        Tool::Select => CursorIcon::Default,
    }
}

/// The edit mode's mouse: turns egui's pointer events over the video into
/// [`EditorInput`]s and returns what they did, merged. Call it only in edit mode:
/// outside it the video must not capture the pointer. (The editor's keys come from
/// `App::keys`.)
///
/// One physical click must give the editor exactly one press. egui reports a short
/// press-and-release without movement as `clicked`, and a press that moves (or is held)
/// as a drag, never both; so `clicked` becomes `Press` + `Release`, and a drag becomes
/// `Press` (at the press origin) + `Drag`… + `Release`.
pub fn interact(ui: &mut egui::Ui, video: Rect, editor: &mut Editor, rules: &mut Rules) -> Outcome {
    let view = viewport(video);
    let response = ui.interact(
        video,
        ui.make_persistent_id("rules_edit"),
        Sense::click_and_drag(),
    );
    let mut outcome = Outcome::default();
    let mut feed = |input: EditorInput| {
        let o = editor.on_input(input, rules, &view);
        outcome.changed |= o.changed;
    };

    let primary = PointerButton::Primary;
    let (shift, press_origin) = ui.input(|i| (i.modifiers.shift, i.pointer.press_origin()));
    let pointer = response.interact_pointer_pos().or(response.hover_pos());
    if !response.dragged_by(primary)
        && let Some(p) = response.hover_pos()
    {
        feed(EditorInput::Hover(to_point(p)));
    }
    if response.drag_started_by(primary)
        && let Some(p) = press_origin
    {
        feed(EditorInput::Press(to_point(p)));
    }
    if response.dragged_by(primary)
        && let Some(p) = pointer
    {
        let pos = to_point(p);
        feed(EditorInput::Drag { pos, snap: shift });
    }
    if response.drag_stopped_by(primary) {
        // The editor ends a drag where the last `Drag` put it.
        feed(EditorInput::Release);
    }
    if response.clicked()
        && let Some(p) = pointer
    {
        feed(EditorInput::Press(to_point(p)));
        feed(EditorInput::Release);
    }
    if response.secondary_clicked()
        && let Some(p) = pointer
    {
        feed(EditorInput::SecondaryClick(to_point(p)));
    }

    set_cursor(ui, &response, editor, rules, view);
    outcome
}

/// Sets the pointer's shape while it is over the video (see [`cursor_for`]).
fn set_cursor(
    ui: &egui::Ui,
    response: &egui::Response,
    editor: &Editor,
    rules: &Rules,
    view: Viewport,
) {
    let dragging = response.dragged_by(PointerButton::Primary);
    if !dragging && !response.hovered() {
        return;
    }
    let hover = response
        .hover_pos()
        .and_then(|p| Editor::hit_test(rules, to_point(p), &view));
    // A drag that started on empty video grabs nothing (it deselected), so it keeps
    // the plain pointer.
    let grabbing = dragging && editor.selected().is_some();
    ui.ctx()
        .set_cursor_icon(cursor_for(editor.tool(), grabbing, hover));
}
