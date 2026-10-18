//! The right-hand side panel: the pipeline strip, one page, and the key hints.
//!
//! The panel has the shape of the pipeline. At the top, the fixed [`pipeline::strip`]
//! lists the stages (Camera → Detect → Track → Events, then Latency), each with its
//! status dot and the one number to watch. Below it, the open stage's **page**: its
//! title, one sentence on what the stage does, and its settings and readouts. Only the
//! page scrolls, so the strip and the key hints pinned at the bottom stay in view.
//!
//! Clicking a row (or pressing 1–5, handled in `App::keys`) changes [`PanelUi::page`].
//! The rows' dots and hover reasons come from [`PanelUi::strip`], which `App` refreshes
//! on its 4 Hz tick (`StripState::new`), so a redraw only reads them.

use eframe::egui::{self, Align, Layout, RichText, ScrollArea};

use super::DashText;
use super::camera_page::{self, CameraPage};
use super::detection_card::{self, DetectionUi};
use super::events_card::{self, CardActions, EventsPanel};
use super::latency_page::{self, LatencyUi};
use super::pipeline::{self, Stage, StripState};
use super::tracking_card::{self, TrackingUi};
use crate::telemetry::TelemetrySnapshot;

/// The global keys. D, T and E are shown on the pages where their controls live.
const FOOTER: &str = "Esc quit · Tab panel · F fullscreen · C clock · 1–5 pages";

/// The panel's own state, owned by `App`.
pub struct PanelUi {
    /// The page shown below the strip.
    pub page: Stage,
    /// The Latency page's Hold and waterfall bars.
    pub latency: LatencyUi,
    /// The strip's dots and their hover reasons, as of `App`'s last 4 Hz tick (right
    /// after the tick's save, so a failed save shows at once).
    pub strip: StripState,
}

impl Default for PanelUi {
    /// Opens on Latency: the stats are what the user looks at first, and the strip
    /// already shows each stage's own number.
    fn default() -> Self {
        Self {
            page: Stage::Latency,
            latency: LatencyUi::default(),
            strip: StripState::default(),
        }
    }
}

/// What the stage pages read and edit, grouped so `show`'s argument list stays short.
pub struct Pages<'a> {
    /// The Detect page's state.
    pub detect: &'a mut DetectionUi,
    /// The Track page's state.
    pub track: &'a mut TrackingUi,
    /// The Events page's rules and totals.
    pub events: EventsPanel<'a>,
    pub camera: CameraPage<'a>,
}

/// The panel. `detection` is false with `--no-detect`. Returns what the Events page
/// did, or nothing on other pages.
pub fn show(
    ui: &mut egui::Ui,
    panel: &mut PanelUi,
    snap: &TelemetrySnapshot,
    text: &DashText,
    pages: Pages,
    detection: bool,
) -> CardActions {
    // Every redraw, whichever page is open, so the bars are current when it opens.
    panel.latency.refresh(snap);
    ui.with_layout(Layout::bottom_up(Align::Min), |ui| {
        ui.add_space(4.0);
        ui.label(RichText::new(FOOTER).small().weak());
        ui.separator();
        ui.with_layout(Layout::top_down(Align::Min), |ui| {
            ui.add_space(8.0);
            let rows = pipeline::rows(text, &panel.strip);
            if let Some(stage) = pipeline::strip(ui, &rows, panel.page) {
                panel.page = stage;
            }
            ui.separator();
            page_header(ui, panel.page);
            // One scroll offset per page, so switching pages doesn't lose your place.
            ScrollArea::vertical()
                .id_salt(panel.page.title())
                .show(ui, |ui| page(ui, panel, snap, text, pages, detection))
                .inner
        })
        .inner
    })
    .inner
}

/// The open page's title and its one-sentence blurb.
fn page_header(ui: &mut egui::Ui, stage: Stage) {
    ui.heading(stage.title());
    ui.label(RichText::new(stage.blurb()).small().weak());
    ui.add_space(4.0);
}

/// The open page's content. Only the Events page reports actions.
fn page(
    ui: &mut egui::Ui,
    panel: &mut PanelUi,
    snap: &TelemetrySnapshot,
    text: &DashText,
    pages: Pages,
    detection: bool,
) -> CardActions {
    let none = CardActions::default();
    match panel.page {
        Stage::Camera => {
            camera_page::show(ui, pages.camera, text);
            none
        }
        Stage::Detect => {
            let view = pages.track.view;
            detection_card::show(ui, pages.detect, view, text, detection);
            none
        }
        Stage::Track => {
            tracking_card::show(ui, pages.track, text, detection);
            none
        }
        Stage::Events => events_card::show(ui, pages.events, snap.events_dropped, detection),
        Stage::Latency => {
            let ep = pages.camera.settings.ep;
            latency_page::show(ui, &mut panel.latency, snap, text, ep);
            none
        }
    }
}
