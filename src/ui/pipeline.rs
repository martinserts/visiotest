//! The pipeline strip: the side panel's menu, in the shape of the pipeline itself.
//!
//! Frames flow Camera → Detect → Track → Events, and the strip lists those stages in
//! that order, linked by a thin line like stops on a subway map. Each row is one stage:
//! its **dot** says whether the stage is healthy, its **number** is the one to watch
//! (camera fps, detector fps, tracks, counts), and its **key** (1–5) opens the stage's
//! page below the strip. Latency sits under a separator: it is not a stage, it measures
//! all of them.
//!
//! A dot that turns amber explains itself: hovering the row shows the **reason**
//! ("Camera at 24.1 fps, below 90 % of 30"); hovering any other row shows the stage's
//! blurb. Each `*_status` function returns the health and its reason together, so the
//! rule and its explanation can't disagree. Reasons are strings, so `App` computes them
//! on its 4 Hz tick into a [`StripState`], never per redraw.
//!
//! Everything here except [`strip`] is pure: the stages, the status rules and the rows
//! are plain data, so they are unit-tested. The row texts come from [`DashText`], which
//! rebuilds them at 4 Hz, so drawing the strip formats nothing.

use eframe::egui::{self, Align2, Color32, Pos2, Rect, Sense, Stroke, TextStyle, vec2};

use super::status_bar::camera_fps_ok;
use super::{AMBER, DashText, GREEN, NEUTRAL, metric_style};
use crate::rules_session::SaveStatus;
use crate::telemetry::TelemetrySnapshot;

/// A row's height: room for the 14 pt name with some air, so a row is easy to click.
const ROW_H: f32 = 28.0;
/// The status dot's radius: big enough to read its colour at a glance, small enough
/// not to compete with the numbers.
const DOT_R: f32 = 4.0;
/// The open row's accent bar, on its left edge.
const ACCENT_W: f32 = 3.0;
/// Where the dot's centre sits, measured from the row's left edge (past the accent bar).
const DOT_X: f32 = 14.0;
/// Where the stage name starts.
const NAME_X: f32 = 28.0;
/// The key hint's column on the right; the value is right-aligned just left of it.
const KEY_W: f32 = 22.0;
/// The gap between the right edge and the key hint.
const RIGHT_PAD: f32 = 8.0;

/// The panel's five pages, in pipeline order. `ALL` drives the strip and keys 1–5.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Stage {
    Camera,
    Detect,
    Track,
    Events,
    Latency,
}

impl Stage {
    /// Every page, in strip order (which is also the order of keys 1–5).
    pub const ALL: [Stage; 5] = [
        Stage::Camera,
        Stage::Detect,
        Stage::Track,
        Stage::Events,
        Stage::Latency,
    ];

    /// The row's name and the page's heading.
    pub fn title(self) -> &'static str {
        match self {
            Stage::Camera => "Camera",
            Stage::Detect => "Detect",
            Stage::Track => "Track",
            Stage::Events => "Events",
            Stage::Latency => "Latency",
        }
    }

    /// One plain sentence on what the stage does, shown under the page's heading.
    pub fn blurb(self) -> &'static str {
        match self {
            Stage::Camera => {
                "Grabs frames from the webcam and shows them. Everything else starts here."
            }
            Stage::Detect => {
                "Finds objects in one frame at a time. It knows nothing about the frame before."
            }
            Stage::Track => {
                "Links detections across frames, so each object keeps its #id, even through \
                 a short occlusion."
            }
            Stage::Events => "Counts tracked objects that cross a line or stay in a zone.",
            Stage::Latency => "Where the time goes, from the camera sensor to the screen.",
        }
    }

    /// The key that opens the page: `'1'` for Camera … `'5'` for Latency.
    pub fn key(self) -> char {
        match self {
            Stage::Camera => '1',
            Stage::Detect => '2',
            Stage::Track => '3',
            Stage::Events => '4',
            Stage::Latency => '5',
        }
    }

    /// Whether this is a pipeline stage. Latency isn't: it measures the stages, so the
    /// strip draws it below a separator and without the connector.
    pub fn is_stage(self) -> bool {
        self != Stage::Latency
    }
}

/// A strip row's status dot.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Health {
    /// The stage doesn't run (`--no-detect`): a hollow grey ring.
    Off,
    /// No data yet: a grey dot.
    #[default]
    Idle,
    /// Running as expected: green.
    Ok,
    /// Running, but something needs a look: amber.
    Warn,
}

/// A row's health and, when it is `Warn`, why (shown on hover).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Status {
    pub health: Health,
    /// One line per problem; `Some` exactly when `health` is `Warn`.
    pub reason: Option<String>,
}

impl Status {
    /// A status with no reason: Off, Idle or Ok.
    fn plain(health: Health) -> Self {
        Self {
            health,
            reason: None,
        }
    }

    /// Ok when nothing is wrong, else Warn with each problem on its own line.
    fn from_problems(problems: Vec<String>) -> Self {
        if problems.is_empty() {
            Self::plain(Health::Ok)
        } else {
            Self {
                health: Health::Warn,
                reason: Some(problems.join("\n")),
            }
        }
    }
}

/// Grey until the first frame; amber when the camera delivers under 90 % of its nominal
/// rate (the status bar's rule) or hands over buffers the capture had to reject.
pub fn camera_status(s: &TelemetrySnapshot, nominal_fps: u32) -> Status {
    if s.seq == 0 {
        return Status::plain(Health::Idle);
    }
    let mut problems = Vec::new();
    if !camera_fps_ok(s.camera_fps, nominal_fps) {
        problems.push(format!(
            "Camera at {:.1} fps, below 90 % of {nominal_fps}",
            s.camera_fps
        ));
    }
    if s.bad_buffers > 0 {
        problems.push(format!("{} bad buffers", s.bad_buffers));
    }
    Status::from_problems(problems)
}

/// Amber on any detector error, or when every frame is asked for and the detector runs
/// at under 90 % of the camera's rate. Why only then: asked for every frame, a slower
/// detector is falling behind; paced (Fixed or Auto), it is slower on purpose.
pub fn detect_status(s: &TelemetrySnapshot, detection: bool) -> Status {
    if !detection {
        return Status::plain(Health::Off);
    }
    if s.detect_total.is_none() {
        return Status::plain(Health::Idle);
    }
    let mut problems = Vec::new();
    if s.det_errors > 0 {
        problems.push(format!("{} detector errors", s.det_errors));
    }
    if falls_behind(s) {
        problems.push(format!(
            "Detector at {:.1} fps, behind the camera's {:.1}",
            s.detector_fps, s.camera_fps
        ));
    }
    Status::from_problems(problems)
}

/// Every frame is asked for, and fewer than 90 % of them get a result.
fn falls_behind(s: &TelemetrySnapshot) -> bool {
    let every_frame = s.detect_interval == Some(std::time::Duration::ZERO);
    every_frame && s.detector_fps < 0.9 * s.camera_fps
}

/// Off and Idle as Detect, else green: the tracker has no failure mode to report.
pub fn track_status(s: &TelemetrySnapshot, detection: bool) -> Status {
    if !detection {
        Status::plain(Health::Off)
    } else if s.detect_total.is_none() {
        Status::plain(Health::Idle)
    } else {
        Status::plain(Health::Ok)
    }
}

/// Amber when the event sink dropped events or the rules file failed to save
/// (`save_error` is the failure's text). There is no Idle: the rules exist (and can be
/// edited) before the first result.
pub fn events_status(s: &TelemetrySnapshot, detection: bool, save_error: Option<&str>) -> Status {
    if !detection {
        return Status::plain(Health::Off);
    }
    let mut problems = Vec::new();
    if s.events_dropped > 0 {
        problems.push(format!("{} events dropped", s.events_dropped));
    }
    if let Some(e) = save_error {
        problems.push(format!("Rules not saved: {e}"));
    }
    Status::from_problems(problems)
}

/// The four stages' statuses, as the strip shows them. `App` rebuilds it on its 4 Hz
/// tick (the reasons are strings) and keeps it in `PanelUi`. Latency has none: no
/// threshold would be honest, so its dot stays neutral.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct StripState {
    pub camera: Status,
    pub detect: Status,
    pub track: Status,
    pub events: Status,
}

impl StripState {
    /// Runs every rule. `nominal_fps` is the camera's advertised rate, `detection` is
    /// false with `--no-detect`, and `save` is the rules file's state (only a failure
    /// counts).
    pub fn new(
        snap: &TelemetrySnapshot,
        nominal_fps: u32,
        detection: bool,
        save: &SaveStatus,
    ) -> Self {
        let save_error = match save {
            SaveStatus::Failed(e) => Some(e.as_str()),
            SaveStatus::Clean | SaveStatus::Saved => None,
        };
        Self {
            camera: camera_status(snap, nominal_fps),
            detect: detect_status(snap, detection),
            track: track_status(snap, detection),
            events: events_status(snap, detection, save_error),
        }
    }
}

/// One strip row as the strip draws it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StripRow<'a> {
    pub stage: Stage,
    /// The live number, e.g. `29.9 fps`, from [`DashText`].
    pub value: &'a str,
    pub health: Health,
    /// Why the dot is amber, shown on hover instead of the blurb.
    pub reason: Option<&'a str>,
}

/// The five rows, in [`Stage::ALL`] order: values from `text`, health and reasons from
/// `strip` (built on the 4 Hz tick). It only copies `&str`s and enums, so calling it on
/// every redraw costs nothing.
pub fn rows<'a>(text: &'a DashText, strip: &'a StripState) -> [StripRow<'a>; 5] {
    let row = |stage, value: &'a String, status: &'a Status| StripRow {
        stage,
        value: value.as_str(),
        health: status.health,
        reason: status.reason.as_deref(),
    };
    // Latency has no health: no threshold would be honest, so it stays neutral.
    let latency = StripRow {
        stage: Stage::Latency,
        value: text.latency_strip.as_str(),
        health: Health::Idle,
        reason: None,
    };
    [
        row(Stage::Camera, &text.camera_strip, &strip.camera),
        row(Stage::Detect, &text.detect_strip, &strip.detect),
        row(Stage::Track, &text.track_strip, &strip.track),
        row(Stage::Events, &text.events_strip, &strip.events),
        latency,
    ]
}

/// Draws the strip and returns the stage that was clicked, if any. `open` is the page
/// shown below, whose row gets the accent bar.
pub fn strip(ui: &mut egui::Ui, rows: &[StripRow; 5], open: Stage) -> Option<Stage> {
    let mut clicked = None;
    // The previous stage's dot, where the connector to this one starts.
    let mut last_dot: Option<Pos2> = None;
    ui.scope(|ui| {
        // Rows touch, so the connector reads as one line and the fills don't gap.
        ui.spacing_mut().item_spacing.y = 0.0;
        for row in rows {
            if !row.stage.is_stage() {
                ui.separator();
            }
            let size = vec2(ui.available_width(), ROW_H);
            let (rect, response) = ui.allocate_exact_size(size, Sense::click());
            let dot = paint_row(ui, rect, row, row.stage == open, response.hovered());
            if row.stage.is_stage() {
                if let Some(above) = last_dot {
                    connector(ui.painter(), above, dot);
                }
                last_dot = Some(dot);
            }
            if response.clicked() {
                clicked = Some(row.stage);
            }
            hover_text(response, row.reason.unwrap_or(row.stage.blurb()));
        }
    });
    clicked
}

/// Shows `text` as the row's tooltip. Through `on_hover_ui`, not `on_hover_text`: the
/// closure runs only while the row is hovered, so other redraws copy no string.
fn hover_text(response: egui::Response, text: &str) {
    response.on_hover_ui(|ui| {
        // Wrap at egui's usual tooltip width, as `on_hover_text` does.
        ui.set_max_width(ui.spacing().tooltip_width);
        ui.label(text);
    });
}

/// Paints one row (fill, accent bar, dot, name, value, key) and returns its dot's centre.
fn paint_row(ui: &egui::Ui, rect: Rect, row: &StripRow, open: bool, hovered: bool) -> Pos2 {
    let visuals = ui.visuals();
    let painter = ui.painter();
    // A lighter fill marks the open page; a fainter one follows the pointer.
    if open {
        painter.rect_filled(rect, 0.0, visuals.widgets.inactive.weak_bg_fill);
    } else if hovered {
        painter.rect_filled(rect, 0.0, visuals.widgets.noninteractive.weak_bg_fill);
    }
    if open {
        // The selection colour, not cyan: cyan is the rules' colour on the video.
        let bar = Rect::from_min_size(rect.min, vec2(ACCENT_W, rect.height()));
        painter.rect_filled(bar, 0.0, visuals.selection.stroke.color);
    }

    let mid = rect.center().y;
    let dot = Pos2::new(rect.left() + DOT_X, mid);
    paint_dot(painter, dot, row.health, visuals.weak_text_color());

    let name_color = if open {
        visuals.strong_text_color()
    } else {
        visuals.text_color()
    };
    let body = TextStyle::Body.resolve(ui.style());
    let name_at = Pos2::new(rect.left() + NAME_X, mid);
    painter.text(
        name_at,
        Align2::LEFT_CENTER,
        row.stage.title(),
        body,
        name_color,
    );

    let key_right = rect.right() - RIGHT_PAD;
    let small = TextStyle::Small.resolve(ui.style());
    let key_at = Pos2::new(key_right, mid);
    let weak = visuals.weak_text_color();
    painter.text(key_at, Align2::RIGHT_CENTER, row.stage.key(), small, weak);

    let metric = metric_style().resolve(ui.style());
    let value_at = Pos2::new(key_right - KEY_W, mid);
    painter.text(
        value_at,
        Align2::RIGHT_CENTER,
        row.value,
        metric,
        name_color,
    );
    dot
}

/// Filled dots for Idle, Ok and Warn; a hollow ring for Off (the stage isn't there).
fn paint_dot(painter: &egui::Painter, at: Pos2, health: Health, weak: Color32) {
    let fill = match health {
        Health::Off => {
            painter.circle_stroke(at, DOT_R, Stroke::new(1.0, weak));
            return;
        }
        Health::Idle => weak,
        Health::Ok => GREEN,
        Health::Warn => AMBER,
    };
    painter.circle_filled(at, DOT_R, fill);
}

/// The thin line from one stage's dot down to the next, so the strip reads as a flow.
/// It stops at the dots' edges, so it never paints over a dot's colour.
fn connector(painter: &egui::Painter, above: Pos2, below: Pos2) {
    let from = above + vec2(0.0, DOT_R);
    let to = below - vec2(0.0, DOT_R);
    // Half-strength grey: a guide for the eye, quieter than the dots and the text.
    painter.line_segment([from, to], Stroke::new(1.0, NEUTRAL.gamma_multiply(0.5)));
}

#[cfg(test)]
mod tests;
