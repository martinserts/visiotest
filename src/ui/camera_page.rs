//! The Camera page: what the webcam delivers and how the window shows it.
//!
//! Everything else in the pipeline starts with these frames, so this is where to look
//! first when the video stutters: the camera's own rate, the frames the UI never took
//! (Dropped), the time between frames, and the two display settings fixed at startup
//! (VSYNC and the YUV → RGB colour matrix). The Clock toggle is here too: the wall
//! clock on the video is what you film to measure glass-to-glass latency.

use eframe::egui::{self, RichText};

use super::{DashText, metric_cell, section, status_bar};
use crate::app::SourceInfo;

/// What the page reads and edits, grouped so the panel's argument list stays short.
pub struct CameraPage<'a> {
    pub info: &'a SourceInfo,
    /// VSYNC and the colour matrix, as the status bar shows them.
    pub settings: &'a status_bar::Settings,
    /// The HUD clock on the video (key **C**).
    pub show_clock: &'a mut bool,
}

/// The page: **Live**, a grid of the camera's numbers, then **Display**, the Clock
/// toggle. The camera has nothing to set while running, so its readouts come first.
pub fn show(ui: &mut egui::Ui, page: CameraPage, text: &DashText) {
    section(ui, "Live");
    egui::Grid::new("camera_grid")
        .num_columns(2)
        .spacing([16.0, 4.0])
        .show(ui, |ui| {
            // Names and settings are words, so they are plain labels; the numbers are
            // monospace cells, so they line up.
            ui.label("Camera");
            ui.label(page.info.name.as_str());
            ui.end_row();
            let numbers = [
                ("Resolution", &text.resolution),
                ("Frames", &text.frames),
                ("Dropped", &text.dropped),
                ("Camera fps", &text.camera_fps),
                ("Display fps", &text.display_fps),
                ("Frame interval", &text.interval),
            ];
            for (name, value) in numbers {
                ui.label(name);
                metric_cell(ui, value);
                ui.end_row();
            }
            ui.label("VSYNC");
            ui.label(if page.settings.vsync { "on" } else { "off" });
            ui.end_row();
            ui.label("Colour matrix");
            ui.label(page.settings.matrix);
            ui.end_row();
        });
    section(ui, "Display");
    ui.horizontal(|ui| {
        ui.toggle_value(page.show_clock, "Clock");
        ui.label(RichText::new("C").small().weak());
    });
}
