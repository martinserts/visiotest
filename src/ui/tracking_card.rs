//! The Track page: which boxes the video shows, whether lost tracks are drawn, how
//! often the detector runs, and the tracker's live numbers. The side panel draws the
//! page's heading.
//!
//! Only the detect interval goes back to the detector: `App` compares
//! [`TrackingUi::interval`] with `DetectorSettings` after the panel has drawn and
//! forwards a change. The view switch and "Show lost" are display-only, like the class
//! filter.
//!
//! **Why pace the detector at all?** Between detector runs the tracker's predictions
//! carry the boxes along, so running YOLO on every frame is often wasted work. Fixed
//! runs it at most every N ms; Auto stretches the wait while the predictions hold up and
//! cuts it when something appears or gets lost (see `detect::interval`).

use std::time::Duration;

use eframe::egui::{self, RichText, Slider};

use super::{DashText, metric_cell, section};
use crate::detect::interval::{DetectInterval, MAX_INTERVAL};

/// The slider's lower bound: one frame at 30 fps. A shorter fixed interval would run on
/// every frame anyway, which is what Every frame is for.
const MIN_FIXED_MS: u32 = 33;
/// The slider's upper bound: the detector's own ceiling, so the two never disagree.
const MAX_FIXED_MS: u32 = MAX_INTERVAL.as_millis() as u32;
/// Where the slider starts when the app was launched with another mode: ten runs a
/// second, a middle ground to explore from.
const DEFAULT_FIXED_MS: u32 = 100;
/// From here on, Fixed runs so rarely that a fast walker may never be confirmed: a new
/// track starts with zero velocity, so by the next run the walker has left its box
/// (the sim benchmark shows it). Auto avoids this by running often when tracks are born.
const SLOW_FIXED_MS: u32 = 250;

/// Which boxes the video shows (key **T**).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BoxView {
    /// The tracker's objects with their ids, predicted to the shown frame.
    Tracks,
    /// The detector's raw boxes, without ids, as of its last run.
    Detections,
}

impl BoxView {
    /// The other view.
    pub fn toggled(self) -> Self {
        match self {
            BoxView::Tracks => BoxView::Detections,
            BoxView::Detections => BoxView::Tracks,
        }
    }
}

/// The "Detect" radio buttons. Fixed's duration lives in [`TrackingUi::fixed_ms`], so the
/// slider keeps its value while another mode is picked.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Pacing {
    EveryFrame,
    Fixed,
    Auto,
}

/// What the card edits, owned by `App`. Plain data, so the logic is testable.
pub struct TrackingUi {
    /// Tracks (default) or raw detections.
    pub view: BoxView,
    /// Draw lost tracks (dimmed). On by default: watching a track coast through
    /// an occlusion is the clearest view of the tracker at work.
    pub show_lost: bool,
    pub pacing: Pacing,
    /// The Fixed slider, `MIN_FIXED_MS..=MAX_FIXED_MS`, kept while another mode is picked.
    pub fixed_ms: u32,
}

impl TrackingUi {
    /// Starts from the interval the detector was launched with (`--detect-interval`).
    pub fn new(interval: DetectInterval) -> Self {
        let (pacing, fixed_ms) = match interval {
            DetectInterval::EveryFrame => (Pacing::EveryFrame, DEFAULT_FIXED_MS),
            DetectInterval::Fixed(d) => (Pacing::Fixed, clamp_ms(d)),
            DetectInterval::Auto => (Pacing::Auto, DEFAULT_FIXED_MS),
        };
        Self {
            view: BoxView::Tracks,
            show_lost: true,
            pacing,
            fixed_ms,
        }
    }

    /// The interval the controls describe.
    pub fn interval(&self) -> DetectInterval {
        match self.pacing {
            Pacing::EveryFrame => DetectInterval::EveryFrame,
            Pacing::Fixed => DetectInterval::Fixed(Duration::from_millis(u64::from(self.fixed_ms))),
            Pacing::Auto => DetectInterval::Auto,
        }
    }

    /// True when Fixed is slow enough to miss fast newcomers (see [`SLOW_FIXED_MS`]).
    pub fn slow_fixed(&self) -> bool {
        self.pacing == Pacing::Fixed && self.fixed_ms >= SLOW_FIXED_MS
    }
}

/// A `--detect-interval` duration as a slider value. The CLI allows 1–500 ms; the
/// slider starts at one 30 fps frame.
fn clamp_ms(d: Duration) -> u32 {
    let ms = u32::try_from(d.as_millis()).unwrap_or(u32::MAX);
    ms.clamp(MIN_FIXED_MS, MAX_FIXED_MS)
}

/// The page. With `--no-detect` (`enabled == false`) it is greyed out and says so.
pub fn show(ui: &mut egui::Ui, state: &mut TrackingUi, text: &DashText, enabled: bool) {
    if !enabled {
        ui.label(RichText::new("Tracking off (--no-detect)").small().weak());
    }
    ui.add_enabled_ui(enabled, |ui| {
        section(ui, "Settings");
        controls(ui, state);
        section(ui, "Live");
        readouts(ui, text);
        ui.label(
            RichText::new(
                "Auto saves compute: it detects less while tracks are predictable, \
                 and more when something appears or is lost.",
            )
            .small()
            .weak(),
        );
    });
}

fn controls(ui: &mut egui::Ui, state: &mut TrackingUi) {
    ui.horizontal(|ui| {
        ui.radio_value(&mut state.view, BoxView::Tracks, "Tracks");
        ui.radio_value(&mut state.view, BoxView::Detections, "Detections");
        ui.label(RichText::new("T").small().weak());
    });
    ui.checkbox(&mut state.show_lost, "Show lost");
    ui.horizontal(|ui| {
        ui.label("Detect");
        ui.radio_value(&mut state.pacing, Pacing::EveryFrame, "Every frame");
        ui.radio_value(&mut state.pacing, Pacing::Fixed, "Fixed");
        ui.radio_value(&mut state.pacing, Pacing::Auto, "Auto");
    });
    let slider = Slider::new(&mut state.fixed_ms, MIN_FIXED_MS..=MAX_FIXED_MS).suffix(" ms");
    ui.add_enabled(state.pacing == Pacing::Fixed, slider);
    if state.slow_fixed() {
        ui.label(
            RichText::new("Slow: a fast newcomer may move out of its box before it's confirmed.")
                .small()
                .weak(),
        );
    }
}

fn readouts(ui: &mut egui::Ui, text: &DashText) {
    egui::Grid::new("tracking_grid")
        .num_columns(2)
        .spacing([16.0, 4.0])
        .show(ui, |ui| {
            let rows = [
                ("Interval now", &text.detect_interval),
                ("Tracks", &text.tracks),
                ("Track IDs", &text.track_ids),
                ("Skipped (paced)", &text.det_paced),
            ];
            for (name, value) in rows {
                ui.label(name);
                metric_cell(ui, value);
                ui.end_row();
            }
        });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tracking_ui_from_interval_round_trips() {
        let fixed = DetectInterval::Fixed(Duration::from_millis(120));
        for interval in [DetectInterval::EveryFrame, fixed, DetectInterval::Auto] {
            assert_eq!(TrackingUi::new(interval).interval(), interval);
        }
        assert_eq!(TrackingUi::new(fixed).fixed_ms, 120);
        assert_eq!(TrackingUi::new(fixed).pacing, Pacing::Fixed);
    }

    #[test]
    fn tracking_ui_fixed_clamps() {
        let fast = TrackingUi::new(DetectInterval::Fixed(Duration::from_millis(10)));
        assert_eq!(fast.fixed_ms, 33);
        let slow = TrackingUi::new(DetectInterval::Fixed(Duration::from_secs(2)));
        assert_eq!(slow.fixed_ms, 500);
    }

    #[test]
    fn tracking_ui_keeps_fixed_ms_across_modes() {
        let mut ui = TrackingUi::new(DetectInterval::Fixed(Duration::from_millis(200)));
        ui.pacing = Pacing::Auto;
        assert_eq!(ui.interval(), DetectInterval::Auto);
        ui.pacing = Pacing::Fixed;
        assert_eq!(
            ui.interval(),
            DetectInterval::Fixed(Duration::from_millis(200))
        );
    }

    #[test]
    fn tracking_ui_slow_fixed_only_for_fixed() {
        let mut ui = TrackingUi::new(DetectInterval::Fixed(Duration::from_millis(250)));
        assert!(ui.slow_fixed());
        ui.fixed_ms = 249;
        assert!(!ui.slow_fixed());
        ui.fixed_ms = 500;
        ui.pacing = Pacing::Auto;
        assert!(!ui.slow_fixed());
    }

    #[test]
    fn tracking_ui_starts_on_tracks_with_lost_shown() {
        let ui = TrackingUi::new(DetectInterval::EveryFrame);
        assert_eq!(ui.view, BoxView::Tracks);
        assert!(ui.show_lost);
    }

    #[test]
    fn box_view_toggles() {
        assert_eq!(BoxView::Tracks.toggled(), BoxView::Detections);
        assert_eq!(BoxView::Detections.toggled(), BoxView::Tracks);
    }
}
