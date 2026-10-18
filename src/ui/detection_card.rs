//! The Detect page: the Boxes toggle, the confidence slider, quick filter buttons and
//! the 12 groups with tri-state checkboxes and live counts, then the detector's live
//! rows (its rate, the frames it skipped while busy, and how far its boxes trail the
//! video). The side panel draws the page's heading.
//!
//! The class filter is **display-only**: it decides which boxes `ui::detections`
//! draws, never what the detector looks for. The confidence threshold is the one value
//! that goes back to the detector; `App` copies [`DetectionUi::conf`] into
//! `DetectorSettings` after the panel has drawn.

use eframe::egui::{
    self, Align, Checkbox, Color32, Layout, RichText, Sense, Slider, Visuals,
    collapsing_header::CollapsingState, vec2,
};

use super::detections::group_color;
use super::tracking_card::BoxView;
use super::{DashText, NEUTRAL, metric_cell, metric_style, section};
use crate::detect::classes::{CLASS_NAMES, ClassCounts, ClassFilter, Group, GroupState};

/// The slider's range. Below 0.10 YOLO11n floods the screen with noise; above 0.90
/// almost nothing survives.
const CONF_RANGE: std::ops::RangeInclusive<f32> = 0.10..=0.90;
const SWATCH: f32 = 10.0;

/// What the card edits, owned by `App`. Plain data, so the logic around it is testable.
pub struct DetectionUi {
    /// Draw boxes at all (key **D**).
    pub show_boxes: bool,
    /// Confidence threshold; `App` forwards changes to the detector.
    pub conf: f32,
    /// Which classes are drawn.
    pub filter: ClassFilter,
    /// Objects per class in the newest result, refreshed at 4 Hz with the panel:
    /// tracked objects in the Tracks view, boxes ≥ `conf` in the Detections view.
    pub counts: ClassCounts,
}

impl DetectionUi {
    /// Starts with boxes on, every class shown and the detector's own threshold.
    pub fn new(conf: f32) -> Self {
        Self {
            show_boxes: true,
            conf,
            filter: ClassFilter::all(),
            counts: ClassCounts::default(),
        }
    }
}

/// The page. With `--no-detect` (`enabled == false`) it is greyed out and says so.
/// `view` is the video's box view, which decides what the counts count.
pub fn show(
    ui: &mut egui::Ui,
    state: &mut DetectionUi,
    view: BoxView,
    text: &DashText,
    enabled: bool,
) {
    if !enabled {
        ui.label(RichText::new("Detection off (--no-detect)").small().weak());
    }
    ui.add_enabled_ui(enabled, |ui| {
        controls(ui, state, view);
        readouts(ui, text);
    });
}

/// The detector's live numbers.
fn readouts(ui: &mut egui::Ui, text: &DashText) {
    section(ui, "Live");
    egui::Grid::new("detector_grid")
        .num_columns(2)
        .spacing([16.0, 4.0])
        .show(ui, |ui| {
            let rows = [
                ("Detector fps", &text.detector_fps),
                ("Skipped (busy)", &text.det_skipped),
                ("Boxes behind video", &text.boxes_behind),
            ];
            for (name, value) in rows {
                ui.label(name);
                metric_cell(ui, value);
                ui.end_row();
            }
        });
}

/// **Settings** (Boxes, Confidence), then **Classes** (the filter and its notes).
fn controls(ui: &mut egui::Ui, state: &mut DetectionUi, view: BoxView) {
    section(ui, "Settings");
    ui.horizontal(|ui| {
        ui.toggle_value(&mut state.show_boxes, "Boxes");
        ui.label(RichText::new("D").small().weak());
    });
    ui.horizontal(|ui| {
        ui.label("Confidence");
        ui.add(Slider::new(&mut state.conf, CONF_RANGE).fixed_decimals(2));
    });
    section(ui, "Classes");
    quick_filter_buttons(ui, &mut state.filter);
    for g in Group::ALL {
        group(ui, state, g);
    }
    ui.label(
        RichText::new("Filter is display-only; the detector always sees all 80 classes.")
            .small()
            .weak(),
    );
    ui.label(RichText::new(counts_note(view)).small().weak());
}

/// The All / None / People buttons. Also in the Events card's class picker.
pub fn quick_filter_buttons(ui: &mut egui::Ui, filter: &mut ClassFilter) {
    ui.horizontal(|ui| {
        if ui.small_button("All").clicked() {
            *filter = ClassFilter::all();
        }
        if ui.small_button("None").clicked() {
            *filter = ClassFilter::none();
        }
        if ui.small_button("People").clicked() {
            *filter = ClassFilter::only(Group::People);
        }
    });
}

/// Group `g`'s tri-state checkbox, with `text` beside it if given. The checkbox only
/// displays the tri-state; [`toggle_group`] decides what a click does.
pub fn group_checkbox(ui: &mut egui::Ui, filter: &mut ClassFilter, g: Group, text: Option<&str>) {
    let gs = filter.group_state(g);
    let mut checked = gs == GroupState::All;
    let check = match text {
        Some(text) => Checkbox::new(&mut checked, text),
        None => Checkbox::without_text(&mut checked),
    };
    if ui
        .add(check.indeterminate(gs == GroupState::Some))
        .changed()
    {
        toggle_group(filter, g);
    }
}

/// What the counts count: they follow the video's box view.
fn counts_note(view: BoxView) -> &'static str {
    match view {
        BoxView::Tracks => "Counts: tracked objects.",
        BoxView::Detections => "Counts: boxes ≥ confidence.",
    }
}

/// One collapsible group: the header row, and per-class rows when expanded.
fn group(ui: &mut egui::Ui, state: &mut DetectionUi, g: Group) {
    // A stable, unique id per group, so egui remembers which groups are expanded.
    let id = ui.make_persistent_id(("det_group", g as u8));
    CollapsingState::load_with_default_open(ui.ctx(), id, false)
        .show_header(ui, |ui| group_header(ui, state, g))
        .body(|ui| {
            for c in g.classes() {
                class_row(ui, state, c);
            }
        });
}

fn group_header(ui: &mut egui::Ui, state: &mut DetectionUi, g: Group) {
    group_checkbox(ui, &mut state.filter, g, None);

    let (swatch, _) = ui.allocate_exact_size(vec2(SWATCH, SWATCH), Sense::hover());
    ui.painter().rect_filled(swatch, 2.0, group_color(g));

    let total = state.counts.group(g);
    let visible = visible_in_group(state.counts, state.filter, g);
    let color = count_color(total > 0, visible > 0, ui.visuals());
    ui.label(RichText::new(g.name()).color(color));
    count_cell(ui, total, visible > 0);
}

fn class_row(ui: &mut egui::Ui, state: &mut DetectionUi, class: u8) {
    ui.horizontal(|ui| {
        let mut on = state.filter.contains(class);
        let n = u32::from(state.counts.class(class));
        let color = count_color(n > 0, on, ui.visuals());
        let name = RichText::new(CLASS_NAMES[usize::from(class)]).color(color);
        if ui.add(Checkbox::new(&mut on, name)).changed() {
            state.filter.set(class, on);
        }
        count_cell(ui, n, on);
    });
}

/// A right-aligned monospace count in the rest of the row.
fn count_cell(ui: &mut egui::Ui, n: u32, shown: bool) {
    let color = count_color(n > 0, shown, ui.visuals());
    ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
        ui.label(
            RichText::new(count_text(n, shown))
                .text_style(metric_style())
                .color(color),
        );
    });
}

/// The tri-state click rule: a click turns the whole group **on**, unless it was
/// already fully on, in which case it turns it off. (Plain checkbox semantics would
/// turn a partly-on group off, which hides the classes you just picked.)
pub fn toggle_group(filter: &mut ClassFilter, g: Group) {
    let on = filter.group_state(g) != GroupState::All;
    filter.set_group(g, on);
}

/// Objects in a group whose class the filter shows, i.e. how many boxes it draws.
fn visible_in_group(counts: ClassCounts, filter: ClassFilter, g: Group) -> u32 {
    g.classes()
        .filter(|&c| filter.contains(c))
        .map(|c| u32::from(counts.class(c)))
        .sum()
}

/// Bright when present and drawn, dimmed when absent, and a neutral grey when present
/// but filtered out, so you can still see that something is hidden.
pub fn count_color(present: bool, shown: bool, visuals: &Visuals) -> Color32 {
    match (present, shown) {
        (false, _) => visuals.weak_text_color(),
        (true, true) => visuals.strong_text_color(),
        (true, false) => NEUTRAL,
    }
}

/// `·` for none, the number when drawn, and `⊘ n` when present but hidden. (`⊘` is in
/// egui's monospace font, Hack, which the metric style uses.)
pub fn count_text(n: u32, shown: bool) -> String {
    match (n, shown) {
        (0, _) => "·".to_owned(),
        (n, true) => n.to_string(),
        (n, false) => format!("⊘ {n}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn toggle_group_all_turns_off() {
        let mut f = ClassFilter::all();
        toggle_group(&mut f, Group::Kitchen);
        assert_eq!(f.group_state(Group::Kitchen), GroupState::None);
        // Other groups are untouched.
        assert_eq!(f.group_state(Group::People), GroupState::All);
    }

    #[test]
    fn toggle_group_some_or_none_turns_on() {
        let mut f = ClassFilter::none();
        f.set(41, true); // cup: Kitchen is now partly on
        toggle_group(&mut f, Group::Kitchen);
        assert_eq!(f.group_state(Group::Kitchen), GroupState::All);

        let mut f = ClassFilter::none();
        toggle_group(&mut f, Group::Food);
        assert_eq!(f.group_state(Group::Food), GroupState::All);
        assert_eq!(f.group_state(Group::Kitchen), GroupState::None);
    }

    #[test]
    fn count_color_three_states() {
        let v = Visuals::dark();
        assert_eq!(count_color(true, true, &v), v.strong_text_color());
        assert_eq!(count_color(false, true, &v), v.weak_text_color());
        assert_eq!(count_color(false, false, &v), v.weak_text_color());
        assert_eq!(count_color(true, false, &v), NEUTRAL);
        assert_ne!(NEUTRAL, v.strong_text_color());
    }

    #[test]
    fn count_text_marks_hidden() {
        assert_eq!(count_text(0, true), "·");
        assert_eq!(count_text(0, false), "·");
        assert_eq!(count_text(3, true), "3");
        assert_eq!(count_text(2, false), "⊘ 2");
    }

    #[test]
    fn detection_card_visible_counts_follow_filter() {
        use crate::detect::{BBox, Detection};
        let det = |class| Detection {
            bbox: BBox::new(0.0, 0.0, 1.0, 1.0),
            class,
            score: 0.5,
        };
        // Two cups and a bowl; the bowl is hidden.
        let counts = ClassCounts::from_detections(&[det(41), det(45), det(41)], 0.35);
        let mut f = ClassFilter::all();
        f.set(45, false);
        assert_eq!(visible_in_group(counts, f, Group::Kitchen), 2);
        assert_eq!(counts.group(Group::Kitchen), 3);
    }
}
