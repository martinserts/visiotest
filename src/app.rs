//! The eframe app: takes the newest camera frame and detection result, uploads the
//! frame, and lays out the dashboard (status bar, side panel, video + HUD). Drawing
//! lives in `ui/`; detection runs on its own thread (`detect::worker`).
//!
//! It also holds a [`RulesSession`]: the working copy of the rules, which the rule
//! editor (on the video, in edit mode) and the Events page change, and the policy for
//! saving it. `App` sends each change on to the detector at once.

use std::error::Error;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use eframe::egui::{self, Color32, Key, Margin, Modifiers, Rect, pos2};

use crate::capture::{Frame, FrameSlot, Source};
use crate::color::ColorMatrix;
use crate::detect::classes::ClassCounts;
use crate::detect::interval::DetectInterval;
use crate::detect::model::{Ep, Model};
use crate::detect::worker::{Detections, Detector, DetectorCounters, DetectorSettings};
use crate::emit::EventSink;
use crate::events::rules::Rules;
use crate::rules_session::RulesSession;
use crate::slot::Latest;
use crate::stats::{Throttle, wall_clock_ms};
use crate::telemetry::{Counters, Shown, Telemetry, TelemetrySnapshot};
use crate::track::tracker::TrackState;
use crate::ui::camera_page::CameraPage;
use crate::ui::detection_card::DetectionUi;
use crate::ui::events_card::{CardActions, EventsPanel};
use crate::ui::pipeline::{Stage, StripState};
use crate::ui::rule_editor::EditorKey;
use crate::ui::rules_overlay::{self, Overlay};
use crate::ui::side_panel::{self, Pages, PanelUi};
use crate::ui::tracking_card::{BoxView, TrackingUi};
use crate::ui::{self, DashText, detections, hud, status_bar, tracks};
use crate::video::VideoTexture;

/// The panel's numbers refresh at 4 Hz; video and clock run at the display rate.
const PANEL_REFRESH: Duration = Duration::from_millis(250);
/// Keys 1–5 open the side panel's pages, in `Stage::ALL` order (see `Stage::key`).
const PAGE_KEYS: [Key; 5] = [Key::Num1, Key::Num2, Key::Num3, Key::Num4, Key::Num5];

/// What the frame source reported at startup.
pub struct SourceInfo {
    pub name: String,
    pub format: String,
    pub width: u32,
    pub height: u32,
    pub fps: u32,
}

/// The frame source (camera or video file) and what feeds the UI from it.
pub struct SourceSetup {
    pub source: Source,
    /// The UI's frame slot.
    pub slot: FrameSlot,
    pub info: SourceInfo,
    /// Buffers the capture callback rejected.
    pub errors: Arc<AtomicU64>,
}

/// What the detector worker needs; `None` with `--no-detect`.
pub struct DetectionSetup {
    /// Loaded and warmed up; moved into the worker thread.
    pub model: Model,
    /// The detector's frame slot, which the capture callback also fills.
    pub frames: FrameSlot,
    pub ep: Ep,
    /// The detect interval to start with (`--detect-interval`).
    pub interval: DetectInterval,
}

/// The lines and zones, and where their events go.
pub struct EventsSetup {
    /// Loaded from `path` at startup (empty if the file didn't exist).
    pub rules: Rules,
    /// The rules file (`--rules`), which the editor saves to.
    pub path: PathBuf,
    /// The thread printing JSON event lines; `None` with `--events off` or
    /// `--no-detect`.
    pub sink: Option<EventSink>,
}

pub struct App {
    /// Kept alive for the app's lifetime: dropping it stops the stream or the ffmpeg process.
    _source: Source,
    slot: FrameSlot,
    video: VideoTexture,
    info: SourceInfo,
    settings: status_bar::Settings,
    last: Option<Shown>,
    capture_errors: Arc<AtomicU64>,
    /// Kept alive for the app's lifetime: dropping it stops and joins the thread, then
    /// the JSON event printer it owns.
    _detector: Option<Detector>,
    /// Events the sink dropped (full queue or a failed write); `None` without a sink.
    events_dropped: Option<Arc<AtomicU64>>,
    /// The rules being edited, the rule editor, and when they are saved.
    session: RulesSession,
    /// A text field (a rule name) or a popup had the keyboard at the end of the last
    /// frame. Read from the *last* frame on purpose: egui clears a field's focus as soon
    /// as Esc arrives, before `keys` runs, so the current state would let that Esc quit.
    keyboard_busy: bool,
    /// The detector's input slot, read for its `dropped()` count (frames skipped busy).
    det_frames: Option<FrameSlot>,
    det_out: Latest<Detections>,
    /// Shared with the worker; the confidence slider writes it.
    det_settings: Arc<DetectorSettings>,
    /// The worker's failed and paced frame counts.
    det_counters: Arc<DetectorCounters>,
    /// The newest result, drawn over the video while it is fresh.
    detections: Option<Detections>,
    /// The Detect page: boxes on/off, confidence, class filter, counts.
    det_ui: DetectionUi,
    /// The Track page: box view, show lost, detect interval.
    track_ui: TrackingUi,
    /// Which page the side panel shows.
    panel: PanelUi,
    telemetry: Telemetry,
    refresh: Throttle,
    snapshot: TelemetrySnapshot,
    text: DashText,
    show_panel: bool,
    show_clock: bool,
}

impl App {
    pub fn new(
        cc: &eframe::CreationContext<'_>,
        cam: SourceSetup,
        matrix: ColorMatrix,
        vsync: bool,
        detection: Option<DetectionSetup>,
        events: EventsSetup,
    ) -> Result<Self, Box<dyn Error + Send + Sync>> {
        let rs = cc
            .wgpu_render_state
            .as_ref()
            .ok_or("eframe started without a wgpu render state")?;
        let video = VideoTexture::new(rs, cam.info.width, cam.info.height, matrix);
        ui::apply_theme(&cc.egui_ctx);
        let settings = status_bar::Settings {
            vsync,
            matrix: match matrix {
                ColorMatrix::Bt709 => "BT.709",
                ColorMatrix::Bt601 => "BT.601",
            },
            ep: detection.as_ref().map(|d| d.ep.chip()),
        };

        let det_out = Latest::new();
        let det_settings = Arc::new(match &detection {
            Some(d) => DetectorSettings::new(d.interval),
            None => DetectorSettings::default(),
        });
        // The detector's first run adopts these (version 1).
        det_settings.set_rules(&events.rules);
        let det_counters = Arc::new(DetectorCounters::default());
        let det_ui = DetectionUi::new(det_settings.conf());
        let track_ui = TrackingUi::new(det_settings.interval());
        let text = DashText::new(&cam.info, detection.is_some());
        let events_dropped = events.sink.as_ref().map(EventSink::dropped);
        let (detector, det_frames) = match detection {
            Some(d) => {
                let detector = Detector::spawn(
                    d.model,
                    matrix,
                    d.frames.clone(),
                    det_out.clone(),
                    det_settings.clone(),
                    det_counters.clone(),
                    events.sink,
                )?;
                (Some(detector), Some(d.frames))
            }
            None => (None, None),
        };
        Ok(Self {
            _source: cam.source,
            slot: cam.slot,
            video,
            text,
            info: cam.info,
            settings,
            last: None,
            capture_errors: cam.errors,
            _detector: detector,
            events_dropped,
            session: RulesSession::new(events.rules, events.path),
            keyboard_busy: false,
            det_frames,
            det_out,
            det_settings,
            det_counters,
            detections: None,
            det_ui,
            track_ui,
            panel: PanelUi::default(),
            telemetry: Telemetry::new(),
            refresh: Throttle::new(PANEL_REFRESH),
            snapshot: TelemetrySnapshot::default(),
            show_panel: true,
            show_clock: true,
        })
    }

    /// Esc / Tab / F / C / D / T / E / 1–5, and in edit mode the editor's keys. Consuming
    /// them keeps Tab from also moving widget focus and a digit from reaching a widget.
    ///
    /// While a rule name is being typed (or a popup is open), none of them apply: the
    /// letters go into the name, Delete and Backspace edit it, Tab moves focus and Esc
    /// just leaves the field (egui's own handling). In edit mode Esc goes to the editor,
    /// which unwinds one level at a time (draft → selection → edit mode), so it never
    /// quits from there.
    fn keys(&mut self, ui: &mut egui::Ui) {
        if self.keyboard_busy {
            return;
        }
        if self.session.editor.is_editing() {
            for key in editor_keys(ui) {
                let s = &mut self.session;
                if s.editor.on_key(key, &mut s.rules).changed {
                    s.changed(&self.det_settings);
                }
            }
        } else if ui.input_mut(|i| i.consume_key(Modifiers::NONE, Key::Escape)) {
            ui.ctx().send_viewport_cmd(egui::ViewportCommand::Close);
        }
        let (tab, f, c, d, t, e, fullscreen) = ui.input_mut(|i| {
            (
                i.consume_key(Modifiers::NONE, Key::Tab),
                i.consume_key(Modifiers::NONE, Key::F),
                i.consume_key(Modifiers::NONE, Key::C),
                i.consume_key(Modifiers::NONE, Key::D),
                i.consume_key(Modifiers::NONE, Key::T),
                i.consume_key(Modifiers::NONE, Key::E),
                i.viewport().fullscreen.unwrap_or(false),
            )
        });
        if tab {
            self.show_panel = !self.show_panel;
        }
        if f {
            ui.ctx()
                .send_viewport_cmd(egui::ViewportCommand::Fullscreen(!fullscreen));
        }
        if c {
            self.show_clock = !self.show_clock;
        }
        if d {
            self.det_ui.show_boxes = !self.det_ui.show_boxes;
        }
        if t {
            self.track_ui.view = self.track_ui.view.toggled();
        }
        // Rules can't fire without the detector, so there is no edit mode then. Entering
        // edit mode opens the Events page, so the rules list is in view while you draw.
        if e && self.text.detection() {
            self.session.editor.toggle_editing();
            if self.session.editor.is_editing() {
                self.panel.page = Stage::Events;
            }
        }
        // A page key also shows the panel, in case Tab had hidden it.
        for (stage, key) in Stage::ALL.into_iter().zip(PAGE_KEYS) {
            if ui.input_mut(|i| i.consume_key(Modifiers::NONE, key)) {
                self.panel.page = stage;
                self.show_panel = true;
            }
        }
    }

    /// Takes the newest camera frame (uploading it) and the newest detection result,
    /// if any arrived since the last redraw. Never blocks: both are `take()`s.
    fn poll_inputs(&mut self, frame: &mut eframe::Frame) {
        if let Some(f) = self.slot.take()
            && let Some(rs) = frame.wgpu_render_state()
        {
            let taken_at = Instant::now();
            self.show_frame(rs, &f, taken_at);
        }
        if let Some(d) = self.det_out.take() {
            self.telemetry.on_detections(&d, Instant::now());
            self.detections = Some(d);
        }
    }

    fn show_frame(&mut self, rs: &eframe::egui_wgpu::RenderState, f: &Frame, taken_at: Instant) {
        if let Some(upload) = self.video.upload(rs, f) {
            let shown = Shown {
                seq: f.seq,
                interval: f.interval,
                camera_fps: f.camera_fps,
                sensor_age: f.sensor_age,
                upload,
                received: f.received,
            };
            self.telemetry.on_frame(&shown, taken_at);
            self.last = Some(shown);
        }
    }

    /// Counters kept outside `Telemetry`, for the 4 Hz snapshot.
    fn counters(&self) -> Counters {
        Counters {
            dropped: self.slot.dropped(),
            bad_buffers: self.capture_errors.load(Ordering::Relaxed),
            det_skipped: self.det_frames.as_ref().map_or(0, |s| s.dropped()),
            det_errors: self.det_counters.errors.load(Ordering::Relaxed),
            det_paced: self.det_counters.paced.load(Ordering::Relaxed),
            events_dropped: self
                .events_dropped
                .as_ref()
                .map_or(0, |d| d.load(Ordering::Relaxed)),
        }
    }

    fn fresh_detections(&self, now: Instant) -> Option<&Detections> {
        fresh(&self.detections, now)
    }

    /// Objects per class for the Detect page: tracked objects in the Tracks view
    /// (Lost ones are hidden guesses, so they don't count), boxes ≥ the slider in the
    /// Detections view. Zero once the result is stale.
    fn class_counts(&self, now: Instant) -> ClassCounts {
        let Some(d) = self.fresh_detections(now) else {
            return ClassCounts::default();
        };
        match self.track_ui.view {
            BoxView::Tracks => {
                let tracked = d.tracks.iter().filter(|v| v.state == TrackState::Tracked);
                ClassCounts::from_classes(tracked.map(|v| v.class))
            }
            // Like the boxes, they apply the slider here (the detector keeps weaker
            // boxes for the tracker).
            BoxView::Detections => ClassCounts::from_detections(&d.boxes, self.det_ui.conf),
        }
    }

    /// The central area: aspect-fit video plus the HUD, clipped to the video rect. In
    /// edit mode the video also takes the mouse for the rule editor.
    fn video_area(&mut self, ui: &mut egui::Ui, now: Instant) {
        let area = ui.max_rect();
        let video = hud::fit_rect(area, self.video.size());
        let painter = ui.painter_at(video);
        match &self.last {
            None => hud::waiting(ui, area),
            Some(last) => {
                let uv = Rect::from_min_max(pos2(0.0, 0.0), pos2(1.0, 1.0));
                painter.image(self.video.texture_id(), video, uv, Color32::WHITE);
                // Boxes above the image but under the HUD, so the clock stays readable.
                if self.det_ui.show_boxes
                    && let Some(d) = self.fresh_detections(now)
                {
                    let (size, filter) = (self.video.size(), self.det_ui.filter);
                    match self.track_ui.view {
                        // Predict-to-display: every track is moved to the time of the
                        // frame on screen (`last.received`), not to `now`, so the box
                        // matches the image even when the detector last ran 0.5 s ago.
                        BoxView::Tracks => {
                            let show_lost = self.track_ui.show_lost;
                            tracks::draw(
                                &painter,
                                video,
                                size,
                                &d.tracks,
                                filter,
                                show_lost,
                                last.received,
                            );
                        }
                        // The slider applies instantly here, at draw time. The detector
                        // also reads it, but as the tracker's high threshold.
                        BoxView::Detections => {
                            let conf = self.det_ui.conf;
                            detections::draw(&painter, video, size, &d.boxes, filter, conf);
                        }
                    }
                }
                // Rules over the boxes, still under the HUD.
                if self.text.detection() {
                    self.draw_rules(&painter, video, last.received, now);
                }
                hud::frame_chip(&painter, video, last.seq);
                if let Some(age) = hud::stalled_for(last.received, now) {
                    hud::stalled(&painter, video, age);
                }
            }
        }
        // Wall clock at frame-build time, just before present. Point the camera at this
        // window: this clock minus its image inside the video is glass-to-glass latency.
        if self.show_clock {
            hud::clock(&painter, video, wall_clock_ms());
        }
        if self.session.editor.is_editing() {
            self.edit_rules(ui, video);
        }
    }

    /// Lines and zones with their counts, dwell timers and (in edit mode) handles.
    fn draw_rules(&self, painter: &egui::Painter, video: Rect, shown_at: Instant, now: Instant) {
        let d = self.fresh_detections(now);
        // Dwell timers hang on the track boxes, so they hide with them (key D).
        let tracks = match d {
            Some(d) if self.det_ui.show_boxes => d.tracks.as_slice(),
            _ => &[],
        };
        let overlay = Overlay {
            rules: &self.session.rules,
            totals: d.map(|d| &d.totals),
            tracks,
            editor: self
                .session
                .editor
                .is_editing()
                .then_some(&self.session.editor),
            shown_at,
            now,
        };
        rules_overlay::draw(painter, video, self.video.size(), &overlay);
    }

    /// Feeds the pointer over the video to the rule editor.
    fn edit_rules(&mut self, ui: &mut egui::Ui, video: Rect) {
        let s = &mut self.session;
        if rules_overlay::interact(ui, video, &mut s.editor, &mut s.rules).changed {
            s.changed(&self.det_settings);
        }
    }
}

/// The rule editor's keys: Esc, Enter, Delete and Backspace, consumed so nothing else
/// acts on them. A Mac's "delete" key is Backspace, so both remove the selected rule.
fn editor_keys(ui: &mut egui::Ui) -> impl Iterator<Item = EditorKey> {
    let (cancel, finish, delete, backspace) = ui.input_mut(|i| {
        (
            i.consume_key(Modifiers::NONE, Key::Escape),
            i.consume_key(Modifiers::NONE, Key::Enter),
            i.consume_key(Modifiers::NONE, Key::Delete),
            i.consume_key(Modifiers::NONE, Key::Backspace),
        )
    });
    [
        (cancel, EditorKey::Cancel),
        (finish, EditorKey::Finish),
        (delete || backspace, EditorKey::Delete),
    ]
    .into_iter()
    .filter_map(|(pressed, key)| pressed.then_some(key))
}

/// The newest detections, unless the detector has produced nothing for
/// [`hud::STALL_AFTER`]: stale boxes must never freeze on screen. A free function (not
/// a method) so the side panel can borrow it next to other fields of `App`.
fn fresh(detections: &Option<Detections>, now: Instant) -> Option<&Detections> {
    detections
        .as_ref()
        .filter(|d| hud::stalled_for(d.done, now).is_none())
}

impl eframe::App for App {
    fn ui(&mut self, ui: &mut egui::Ui, frame: &mut eframe::Frame) {
        let now = Instant::now();
        self.telemetry.on_ui(now);
        self.poll_inputs(frame);
        if self.refresh.due(now) {
            self.snapshot = self.telemetry.snapshot(self.counters());
            self.text.refresh(&self.snapshot);
            let (rules, editing) = (self.session.rules.len(), self.session.editor.is_editing());
            let totals = fresh(&self.detections, now).map(|d| &d.totals);
            self.text.refresh_events(rules, totals, editing);
            // Counts refresh with the panel numbers, so they read calmly at 4 Hz too.
            self.det_ui.counts = self.class_counts(now);
            self.session.tick();
            let (snap, fps, save) = (&self.snapshot, self.info.fps, self.session.status());
            self.panel.strip = StripState::new(snap, fps, self.text.detection(), save);
        }
        self.keys(ui);

        // Panels before the central panel, or it takes the whole window.
        let bar = egui::Frame::side_top_panel(ui.style()).inner_margin(Margin::symmetric(8, 6));
        egui::Panel::top("status").frame(bar).show(ui, |ui| {
            status_bar::show(ui, &self.info, &self.settings, &self.snapshot, &self.text);
        });
        let mut card = CardActions::default();
        // 360 px fits the latency table's three 7-char columns plus the stage names.
        egui::Panel::right("side")
            .resizable(true)
            .default_size(360.0)
            .size_range(320.0..=520.0)
            .show_collapsible(ui, &mut self.show_panel, |ui| {
                let (text, snap) = (&self.text, &self.snapshot);
                let pages = Pages {
                    detect: &mut self.det_ui,
                    track: &mut self.track_ui,
                    events: EventsPanel {
                        session: &mut self.session,
                        totals: fresh(&self.detections, now).map(|d| &d.totals),
                    },
                    camera: CameraPage {
                        info: &self.info,
                        settings: &self.settings,
                        show_clock: &mut self.show_clock,
                    },
                };
                let detection = text.detection();
                card = side_panel::show(ui, &mut self.panel, snap, text, pages, detection);
            });
        // Events page edits are applied here, before the video draws them.
        if card.changed {
            self.session.changed(&self.det_settings);
        }
        if card.reset {
            self.det_settings.request_reset();
        }
        // The slider edits `det_ui.conf`; only the threshold goes back to the detector,
        // where it is the tracker's high threshold.
        // (The class filter is display-only and is read by `video_area` below.)
        if self.det_ui.conf != self.det_settings.conf() {
            self.det_settings.set_conf(self.det_ui.conf);
        }
        // The Track page's Detect controls: the detector picks a changed interval up
        // on its next frame. (The box view and "Show lost" are display-only.)
        let wanted = self.track_ui.interval();
        if wanted != self.det_settings.interval() {
            self.det_settings.set_interval(wanted);
        }
        egui::CentralPanel::no_frame()
            .frame(egui::Frame::NONE.fill(Color32::BLACK))
            .show(ui, |ui| self.video_area(ui, now));

        // For the next frame's `keys` (see `keyboard_busy`).
        let ctx = ui.ctx();
        self.keyboard_busy = ctx.text_edit_focused() || ctx.any_popup_open();

        // Continuous redraw: without it egui sleeps until input and the clock freezes.
        ctx.request_repaint();
    }

    /// One last save, so an edit made less than 250 ms before closing isn't lost.
    fn on_exit(&mut self) {
        if let Some(e) = self.session.flush() {
            eprintln!("rules: not saved: {e}");
        }
    }
}
