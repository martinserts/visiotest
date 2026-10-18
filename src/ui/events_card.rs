//! The Events page: under **Rules**, edit mode, the drawing tools, one row per rule
//! (or, with none yet, Draw a line / Draw a zone buttons) and Reset counts; then whether
//! the rules are saved. The side panel draws the page's heading.
//!
//! Each row shows a rule's name (editable), its counts, and its settings: the box
//! anchor that does the crossing (Feet or Centre), the classes it counts, Flip for a
//! line (swaps which way is A→B) and the dwell time for a zone. Clicking a row in edit
//! mode selects the rule, as clicking it on the video does.
//!
//! The card edits the session's working copy of the rules in place and reports what it
//! did ([`CardActions`]); `App` then calls `RulesSession::changed` (which sends the
//! rules to the detector and marks them for saving) and asks for a reset. Like the
//! editor, the card never does those itself.

use std::path::Path;

use eframe::egui::{
    self, ComboBox, PopupCloseBehavior, RichText, Sense, Slider, Stroke, TextEdit, UiBuilder,
};

use super::detection_card::{group_checkbox, quick_filter_buttons};
use super::rule_editor::{Editor, Tool};
use super::rules_overlay::{line_count_text, zone_text};
use super::{AMBER, metric_style, section};
use crate::detect::classes::{ClassFilter, Group};
use crate::events::event::EventTotals;
use crate::events::rules::{Anchor, LineRule, MAX_DWELL, MIN_DWELL, RuleId, ZoneRule};
use crate::rules_session::{RulesSession, SaveStatus};

/// Long enough for "front door", short enough for the row and a JSON line.
const NAME_CHARS: usize = 24;
const NAME_WIDTH: f32 = 120.0;
const FLIP_TIP: &str = "Swaps which way is A→B; counts so far keep their old direction.";

/// What the card did this frame, for `App` to act on.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CardActions {
    /// The rules were edited (name, anchor, classes, flip, dwell, delete).
    pub changed: bool,
    /// Reset counts was clicked.
    pub reset: bool,
}

/// Everything the card reads and edits, grouped so the panel's argument list stays short.
pub struct EventsPanel<'a> {
    /// The rules, the editor and the save status.
    pub session: &'a mut RulesSession,
    /// The newest fresh totals; `None` without a (fresh) result.
    pub totals: Option<&'a EventTotals>,
}

/// The class picker's title: `All`, `None`, a group's name when exactly that group is
/// on, else how many classes are.
pub fn class_summary(filter: ClassFilter) -> String {
    if filter == ClassFilter::all() {
        return "All".to_owned();
    }
    if filter == ClassFilter::none() {
        return "None".to_owned();
    }
    if let Some(g) = Group::ALL
        .into_iter()
        .find(|&g| filter == ClassFilter::only(g))
    {
        return g.name().to_owned();
    }
    let all_classes = Group::ALL.into_iter().flat_map(Group::classes);
    let n = all_classes.filter(|&c| filter.contains(c)).count();
    if n == 1 {
        "1 class".to_owned()
    } else {
        format!("{n} classes")
    }
}

/// The anchor picker's words: "Feet" is the box's bottom centre.
pub fn anchor_name(anchor: Anchor) -> &'static str {
    match anchor {
        Anchor::BottomCenter => "Feet",
        Anchor::Center => "Centre",
    }
}

/// The save line.
pub fn save_text(save: &SaveStatus, file: &Path) -> String {
    match save {
        SaveStatus::Clean => format!("Rules file: {}", file.display()),
        SaveStatus::Saved => format!("Saved to {}", file.display()),
        SaveStatus::Failed(e) => format!("Not saved: {e}"),
    }
}

/// The page. With `--no-detect` (`enabled == false`) it is greyed out and says so.
/// `dropped` is how many events the printer dropped (0 without one).
pub fn show(ui: &mut egui::Ui, panel: EventsPanel, dropped: u64, enabled: bool) -> CardActions {
    if !enabled {
        ui.label(
            RichText::new("Detection off (--no-detect): rules can't fire.")
                .small()
                .weak(),
        );
    }
    ui.add_enabled_ui(enabled, |ui| controls(ui, panel, dropped))
        .inner
}

fn controls(ui: &mut egui::Ui, p: EventsPanel, dropped: u64) -> CardActions {
    // One split borrow: the rows edit `s.rules` while they also select in `s.editor`.
    let s = &mut *p.session;
    section(ui, "Rules");
    edit_bar(ui, &mut s.editor);
    let mut actions = CardActions::default();
    let mut remove: Option<RuleId> = None;

    for line in &mut s.rules.lines {
        let counts = p.totals.and_then(|t| t.rule(line.id)).map(line_count_text);
        rule_row(
            ui,
            line.id,
            &mut s.editor,
            &mut actions,
            &mut remove,
            |ui| line_row(ui, line, counts),
        );
    }
    for zone in &mut s.rules.zones {
        let counts = p.totals.and_then(|t| t.rule(zone.id)).map(zone_text);
        rule_row(
            ui,
            zone.id,
            &mut s.editor,
            &mut actions,
            &mut remove,
            |ui| zone_row(ui, zone, counts),
        );
    }
    // Removed after the loops, which borrow the rule lists.
    if let Some(id) = remove {
        s.editor.remove(id, &mut s.rules);
        actions.changed = true;
    }

    if s.rules.is_empty() {
        empty_state(ui, &mut s.editor);
    }
    actions.reset = ui.button("Reset counts").clicked();
    footer(ui, s.status(), s.path(), dropped);
    actions
}

/// With no rules yet, the first step as two buttons. They only enter edit mode with a
/// tool, so they don't change the rules (no `actions.changed`). Drawn outside the rows,
/// so the rows' click sense (`UiBuilder::sense`) is untouched.
fn empty_state(ui: &mut egui::Ui, editor: &mut Editor) {
    ui.horizontal(|ui| {
        if ui.button("Draw a line").clicked() {
            start_drawing(editor, Tool::Line);
        }
        if ui.button("Draw a zone").clicked() {
            start_drawing(editor, Tool::Zone);
        }
    });
    ui.label(
        RichText::new("Then drag on the video. E toggles edit mode.")
            .small()
            .weak(),
    );
}

/// Enters edit mode (if not already in it) with `tool` picked. `set_tool` drops any
/// half-drawn shape, as switching tools by hand does.
fn start_drawing(editor: &mut Editor, tool: Tool) {
    if !editor.is_editing() {
        editor.toggle_editing();
    }
    editor.set_tool(tool);
}

/// One rule's row plus the bookkeeping every row shares: the selected highlight,
/// click-to-select (the editor ignores it outside edit mode), and remembering a 🗑
/// press (removed after the loops, which borrow the rule lists).
fn rule_row(
    ui: &mut egui::Ui,
    id: RuleId,
    editor: &mut Editor,
    actions: &mut CardActions,
    remove: &mut Option<RuleId>,
    add: impl FnOnce(&mut egui::Ui) -> RowEdit,
) {
    let selected = editor.selected() == Some(id);
    let (edit, clicked) = row(ui, selected, add);
    actions.changed |= edit.changed;
    if edit.remove {
        *remove = Some(id);
    }
    if clicked {
        editor.select(id);
    }
}

/// The Edit toggle and the tool buttons (usable only in edit mode).
fn edit_bar(ui: &mut egui::Ui, editor: &mut Editor) {
    ui.horizontal(|ui| {
        // A copy for the widget: a click goes through `toggle_editing`, which also
        // resets the tool and the selection when leaving.
        let mut editing = editor.is_editing();
        if ui.toggle_value(&mut editing, "✏ Edit").changed() {
            editor.toggle_editing();
        }
        ui.label(RichText::new("E").small().weak());
    });
    ui.add_enabled_ui(editor.is_editing(), |ui| {
        ui.horizontal(|ui| {
            // A copy, so a change goes through `set_tool`, which drops a half-drawn shape.
            let mut tool = editor.tool();
            ui.selectable_value(&mut tool, Tool::Select, "Select");
            ui.selectable_value(&mut tool, Tool::Line, "Line");
            ui.selectable_value(&mut tool, Tool::Zone, "Zone");
            if tool != editor.tool() {
                editor.set_tool(tool);
            }
        });
    });
}

/// What a row's widgets did.
#[derive(Clone, Copy, Debug, Default)]
struct RowEdit {
    changed: bool,
    remove: bool,
}

/// One rule's framed row (amber outline when selected), and whether its background was
/// clicked. The click sense is registered *under* the row's widgets (`UiBuilder::sense`),
/// so a click on the name field or a button goes to that widget, not to the row.
fn row(
    ui: &mut egui::Ui,
    selected: bool,
    add: impl FnOnce(&mut egui::Ui) -> RowEdit,
) -> (RowEdit, bool) {
    let mut frame = egui::Frame::group(ui.style());
    if selected {
        frame = frame.stroke(Stroke::new(1.5, AMBER));
    }
    let r = ui.scope_builder(UiBuilder::new().sense(Sense::click()), |ui| {
        frame
            .show(ui, |ui| {
                ui.set_width(ui.available_width());
                add(ui)
            })
            .inner
    });
    (r.inner, r.response.clicked())
}

fn line_row(ui: &mut egui::Ui, line: &mut LineRule, counts: Option<String>) -> RowEdit {
    let mut edit = RowEdit::default();
    edit.changed |= name_and_kind(ui, &mut line.name, "line");
    counts_label(ui, counts);
    ui.horizontal(|ui| {
        edit.changed |= anchor_picker(ui, line.id, &mut line.anchor);
        edit.changed |= class_picker(ui, line.id, &mut line.classes);
        if ui.small_button("⇄").on_hover_text(FLIP_TIP).clicked() {
            line.flip();
            edit.changed = true;
        }
        edit.remove = delete_button(ui);
    });
    edit
}

fn zone_row(ui: &mut egui::Ui, zone: &mut ZoneRule, counts: Option<String>) -> RowEdit {
    let mut edit = RowEdit::default();
    edit.changed |= name_and_kind(ui, &mut zone.name, "zone");
    counts_label(ui, counts);
    ui.horizontal(|ui| {
        edit.changed |= anchor_picker(ui, zone.id, &mut zone.anchor);
        edit.changed |= class_picker(ui, zone.id, &mut zone.classes);
        edit.remove = delete_button(ui);
    });
    ui.horizontal(|ui| {
        ui.label("Dwell after");
        // Whole seconds: the file allows fractions, but a slider of them is fiddly.
        let mut secs = zone.dwell_after.as_secs();
        let range = MIN_DWELL.as_secs()..=MAX_DWELL.as_secs();
        if ui.add(Slider::new(&mut secs, range).suffix(" s")).changed() {
            zone.dwell_after = std::time::Duration::from_secs(secs);
            edit.changed = true;
        }
    });
    edit
}

/// The editable name and a weak "line" / "zone".
fn name_and_kind(ui: &mut egui::Ui, name: &mut String, kind: &str) -> bool {
    ui.horizontal(|ui| {
        let field = TextEdit::singleline(name)
            .char_limit(NAME_CHARS)
            .desired_width(NAME_WIDTH);
        let changed = ui.add(field).changed();
        ui.label(RichText::new(kind).small().weak());
        changed
    })
    .inner
}

fn counts_label(ui: &mut egui::Ui, counts: Option<String>) {
    match counts {
        Some(text) => ui.label(RichText::new(text).text_style(metric_style())),
        None => ui.label(RichText::new("no counts yet").small().weak()),
    };
}

fn anchor_picker(ui: &mut egui::Ui, id: RuleId, anchor: &mut Anchor) -> bool {
    let before = *anchor;
    ComboBox::from_id_salt(("rule_anchor", id))
        .selected_text(anchor_name(*anchor))
        .width(70.0)
        .show_ui(ui, |ui| {
            for a in [Anchor::BottomCenter, Anchor::Center] {
                ui.selectable_value(anchor, a, anchor_name(a));
            }
        })
        .response
        .on_hover_text("Which point of a box does the crossing: Feet = bottom centre.");
    *anchor != before
}

/// The rule's classes: quick buttons and the 12 groups with tri-state checkboxes. The
/// popup stays open while you tick, and closes on a click outside it.
fn class_picker(ui: &mut egui::Ui, id: RuleId, classes: &mut ClassFilter) -> bool {
    let before = *classes;
    ComboBox::from_id_salt(("rule_classes", id))
        .selected_text(class_summary(*classes))
        .close_behavior(PopupCloseBehavior::CloseOnClickOutside)
        .show_ui(ui, |ui| {
            quick_filter_buttons(ui, classes);
            for g in Group::ALL {
                group_checkbox(ui, classes, g, Some(g.name()));
            }
        });
    *classes != before
}

fn delete_button(ui: &mut egui::Ui) -> bool {
    ui.small_button("🗑")
        .on_hover_text("Delete this rule")
        .clicked()
}

/// The save line and, if the event printer fell behind, how many events it dropped.
fn footer(ui: &mut egui::Ui, save: &SaveStatus, file: &Path, dropped: u64) {
    let line = save_text(save, file);
    if save.is_failed() {
        ui.label(RichText::new(line).small().color(AMBER));
    } else {
        ui.label(RichText::new(line).small().weak());
    }
    if dropped > 0 {
        ui.label(
            RichText::new(format!("Events dropped: {dropped}"))
                .small()
                .color(AMBER),
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn events_card_class_summary() {
        assert_eq!(class_summary(ClassFilter::all()), "All");
        assert_eq!(class_summary(ClassFilter::none()), "None");
        assert_eq!(class_summary(ClassFilter::only(Group::People)), "People");
        let mut three = ClassFilter::none();
        for class in [0, 2, 41] {
            three.set(class, true);
        }
        assert_eq!(class_summary(three), "3 classes");
        let mut one = ClassFilter::none();
        one.set(2, true); // car: one class, not the whole Vehicles group
        assert_eq!(class_summary(one), "1 class");
    }

    #[test]
    fn events_card_start_drawing_enters_edit_mode_with_the_tool() {
        let mut editor = Editor::new();
        start_drawing(&mut editor, Tool::Line);
        assert!(editor.is_editing());
        assert_eq!(editor.tool(), Tool::Line);
    }

    #[test]
    fn events_card_start_drawing_switches_tool_and_drops_the_draft() {
        use crate::events::geometry::Point;
        use crate::events::rules::Rules;
        use crate::ui::rule_editor::{EditorInput, Viewport};

        let (mut editor, mut rules) = (Editor::new(), Rules::default());
        start_drawing(&mut editor, Tool::Zone);
        let view = Viewport {
            origin: Point { x: 0.0, y: 0.0 },
            width: 640.0,
            height: 360.0,
        };
        // One corner of a zone: a draft, not a rule yet.
        let corner = EditorInput::Press(Point { x: 100.0, y: 100.0 });
        editor.on_input(corner, &mut rules, &view);
        assert!(editor.draft().is_some());

        start_drawing(&mut editor, Tool::Line);
        assert!(editor.is_editing());
        assert_eq!(editor.tool(), Tool::Line);
        assert!(editor.draft().is_none());
        assert!(rules.is_empty());
    }

    #[test]
    fn events_card_save_text() {
        let file = Path::new("rules.json");
        assert_eq!(
            save_text(&SaveStatus::Clean, file),
            "Rules file: rules.json"
        );
        assert_eq!(save_text(&SaveStatus::Saved, file), "Saved to rules.json");
        let failed = SaveStatus::Failed("disk full".into());
        assert_eq!(save_text(&failed, file), "Not saved: disk full");
        // The footer shows only a failure as a warning.
        assert!(failed.is_failed());
        assert!(!SaveStatus::Saved.is_failed());
        assert!(!SaveStatus::Clean.is_failed());
    }
}
