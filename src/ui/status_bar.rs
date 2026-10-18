//! Top status bar: app + camera on the left, status chips on the right.

use eframe::egui::{self, Align, Layout, RichText};

use super::{AMBER, DashText, GREEN, NEUTRAL, chip};
use crate::app::SourceInfo;
use crate::telemetry::TelemetrySnapshot;

/// Static settings the chips show.
pub struct Settings {
    pub vsync: bool,
    /// Colour matrix label, e.g. `BT.709`.
    pub matrix: &'static str,
    /// Execution-provider chip, e.g. `EP CoreML`; `None` with `--no-detect`.
    pub ep: Option<&'static str>,
}

/// Camera fps counts as healthy at ≥ 90 % of the nominal rate.
pub fn camera_fps_ok(fps: f64, nominal: u32) -> bool {
    fps >= 0.9 * f64::from(nominal)
}

pub fn show(
    ui: &mut egui::Ui,
    info: &SourceInfo,
    settings: &Settings,
    snap: &TelemetrySnapshot,
    text: &DashText,
) {
    ui.horizontal_centered(|ui| {
        ui.label(RichText::new("visiotest").strong());
        ui.weak("·");
        ui.label(info.name.as_str());
        ui.weak(text.source.as_str());

        ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
            // Right to left: the first chip is the rightmost.
            chip(ui, &text.display_chip, NEUTRAL);
            let cam = if camera_fps_ok(snap.camera_fps, info.fps) {
                GREEN
            } else {
                AMBER
            };
            chip(ui, &text.camera_chip, cam);
            chip(ui, settings.matrix, NEUTRAL);
            if settings.vsync {
                chip(ui, "VSYNC ON", NEUTRAL);
            } else {
                chip(ui, "VSYNC OFF", AMBER);
            }
            match settings.ep {
                Some(ep) => chip(ui, ep, NEUTRAL),
                None => chip(ui, "DETECT OFF", NEUTRAL),
            }
            if snap.bad_buffers > 0 {
                chip(ui, "BAD BUFFERS", AMBER);
            }
            if snap.det_errors > 0 {
                chip(ui, "DETECT ERRORS", AMBER);
            }
        });
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn camera_fps_threshold() {
        assert!(camera_fps_ok(27.0, 30));
        assert!(!camera_fps_ok(26.9, 30));
        assert!(camera_fps_ok(0.0, 0));
    }
}
