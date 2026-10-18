//! Dashboard drawing: theme, small shared widgets, and (in [`text`]) the panel text cache.
//!
//! Nothing in `ui/` touches capture or GPU state: it reads a `TelemetrySnapshot`, the
//! `SourceInfo` and a [`DashText`] and draws them. The one thing it edits is `App`'s
//! working copy of the rules ([`rule_editor`] on the video, [`events_card`] in the
//! side panel), and it only reports changes; `App` sends and saves them.
//!
//! The side panel ([`side_panel`]) is the [`pipeline`] strip plus one page per stage:
//! [`camera_page`], the Detect, Track and Events pages (the `*_card` modules) and
//! [`latency_page`].

pub mod camera_page;
pub mod detection_card;
pub mod detections;
pub mod events_card;
pub mod hud;
pub mod latency_page;
pub mod pipeline;
pub mod rule_editor;
pub mod rules_overlay;
pub mod side_panel;
pub mod status_bar;
mod text;
pub mod tracking_card;
pub mod tracks;

pub use text::{
    DETECT_TOTAL, DashText, EVENTS, INFER, POST, PRE, SENSOR, TOTAL, TRACK, UPLOAD, WAIT,
};

use eframe::egui::{
    self, Color32, CornerRadius, FontId, Margin, RichText, Stroke, TextStyle, Visuals, vec2,
};

pub const AMBER: Color32 = Color32::from_rgb(0xf0, 0xb4, 0x29);
pub const GREEN: Color32 = Color32::from_rgb(0x3c, 0xcf, 0x91);
pub const NEUTRAL: Color32 = Color32::from_gray(150);
const PANEL_FILL: Color32 = Color32::from_rgb(0x16, 0x18, 0x1c);
/// Space above a [`section`] heading: a little more than the 6 pt between widgets, so a
/// new section reads as a break without a separator line.
const SECTION_SPACE: f32 = 8.0;

/// `"metric"`: monospace numbers in the panel.
pub fn metric_style() -> TextStyle {
    TextStyle::Name("metric".into())
}

/// Dark visuals with the dashboard's spacing, fills and text sizes. Call once at startup.
pub fn apply_theme(ctx: &egui::Context) {
    ctx.set_fonts(fonts_with_symbol_fallback());
    ctx.set_theme(egui::Theme::Dark);
    ctx.style_mut_of(egui::Theme::Dark, |style| {
        style.visuals = Visuals::dark();
        style.visuals.panel_fill = PANEL_FILL;
        style.spacing.item_spacing = vec2(8.0, 6.0);
        let ts = &mut style.text_styles;
        ts.insert(TextStyle::Body, FontId::proportional(14.0));
        ts.insert(TextStyle::Button, FontId::proportional(14.0));
        ts.insert(TextStyle::Small, FontId::proportional(12.0));
        ts.insert(TextStyle::Heading, FontId::proportional(16.0));
        ts.insert(TextStyle::Monospace, FontId::monospace(14.0));
        ts.insert(metric_style(), FontId::monospace(14.0));
    });
}

/// egui's default fonts with Hack (the monospace font) appended as the last fallback
/// of the proportional family. Ubuntu-Light, the proportional font, has no `→` or `⊘`,
/// so labels like "Frame → boxes" would show a missing-glyph box; Hack has both.
/// Icons like ✏ and 🗑 come from egui's emoji fonts (the `monochrome_emoji_fonts`
/// feature in `Cargo.toml`), which egui already puts in every family's fallbacks.
fn fonts_with_symbol_fallback() -> egui::FontDefinitions {
    let mut fonts = egui::FontDefinitions::default();
    if let Some(family) = fonts.families.get_mut(&egui::FontFamily::Proportional) {
        family.push("Hack".to_owned());
    }
    fonts
}

/// A pill: rounded, 1 px stroke, small text.
pub fn chip(ui: &mut egui::Ui, text: &str, color: Color32) {
    egui::Frame::new()
        .stroke(Stroke::new(1.0, color))
        .corner_radius(CornerRadius::same(10))
        .inner_margin(Margin::symmetric(8, 1))
        .show(ui, |ui| ui.label(RichText::new(text).small().color(color)));
}

/// A page's sub-heading ("Settings", "Live"): small, strong, with a little space
/// above, so every page reads in the same order without a heavy rule. A plain label,
/// not a collapsing header: hiding controls is what the pages are meant to avoid.
pub fn section(ui: &mut egui::Ui, title: &str) {
    ui.add_space(SECTION_SPACE);
    ui.label(RichText::new(title).small().strong());
}

/// A monospace number cell (pre-padded to [`text::CELL`] chars, so it reads right-aligned).
pub fn metric_cell(ui: &mut egui::Ui, text: &str) {
    ui.label(RichText::new(text).text_style(metric_style()));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fonts_proportional_falls_back_to_hack() {
        let fonts = fonts_with_symbol_fallback();
        let family = &fonts.families[&egui::FontFamily::Proportional];
        assert_eq!(family.last().map(String::as_str), Some("Hack"));
        assert!(fonts.font_data.contains_key("Hack"));
    }

    #[test]
    fn fonts_include_the_emoji_fonts_for_icons() {
        let fonts = fonts_with_symbol_fallback();
        let family = &fonts.families[&egui::FontFamily::Proportional];
        // emoji-icon-font has the card's ✏ (U+270F) and 🗑 (U+1F5D1).
        assert!(fonts.font_data.contains_key("emoji-icon-font"));
        assert!(family.iter().any(|name| name == "emoji-icon-font"));
    }
}
