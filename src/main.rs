//! visiotest: webcam frames in, converted YUYV → RGB on the GPU and shown in an
//! egui window with per-stage latency, while a worker thread runs YOLO11n (ONNX
//! Runtime, CPU or CoreML) on the newest frame, paced by a detect interval, links the
//! boxes into tracks with stable IDs (a hand-written ByteTrack) and turns the tracks
//! into line-crossing and zone events (`rules.json`), printed as JSON lines on stdout.
//!
//! **stdout carries only event lines**, so `cargo run --release | jq` works; every
//! other message goes to stderr.

mod app;
mod capture;
mod color;
mod detect;
mod emit;
mod events;
mod rules_session;
mod slot;
mod stats;
mod telemetry;
mod track;
mod ui;
mod video;

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::AtomicU64;
use std::sync::mpsc;
use std::time::{Duration, Instant};

use anyhow::{Context, Result, anyhow, bail};
use clap::Parser;
use eframe::{egui, wgpu};
use nokhwa::Camera;
use nokhwa::pixel_format::RgbFormat;
use nokhwa::utils::{ApiBackend, RequestedFormat, RequestedFormatType};

use app::{App, DetectionSetup, EventsSetup, SourceInfo, SourceSetup};
use capture::{FrameSlot, Source};
use color::ColorMatrix;
use detect::interval::DetectInterval;
use detect::model::{Ep, Model};
use emit::{EventSink, EventsOut, WallClock};
use events::rules::Rules;

#[derive(Parser)]
#[command(about = "Webcam → YOLO → tracking → line and zone events, with per-stage latency")]
struct Args {
    /// List cameras and their supported formats, then exit.
    #[arg(long)]
    list: bool,

    /// Camera index (see --list).
    #[arg(long, default_value_t = 0)]
    camera: u32,

    /// Play a video file instead of the camera (looped, decoded by ffmpeg).
    #[arg(long, value_name = "FILE", conflicts_with = "camera")]
    video: Option<PathBuf>,

    /// Disable vsync (lowest latency, may tear).
    #[arg(long)]
    no_vsync: bool,

    /// YUV → RGB colour matrix (limited range).
    #[arg(long, value_enum, default_value_t = ColorMatrix::Bt709)]
    matrix: ColorMatrix,

    /// ONNX model (export it with model-export/export.py).
    #[arg(long, default_value = "models/yolo11n.onnx")]
    model: PathBuf,

    /// ONNX Runtime execution provider.
    #[arg(long, value_enum, default_value_t = Ep::Cpu)]
    ep: Ep,

    /// When the detector runs: every (frame), auto, or a fixed interval in ms (1–500).
    #[arg(long, default_value = "every")]
    detect_interval: DetectInterval,

    /// Run without detection: video only, no model loaded.
    #[arg(long)]
    no_detect: bool,

    /// Lines and zones (drawn with E in the window, saved here).
    #[arg(long, default_value = "rules.json")]
    rules: PathBuf,

    /// Where events go.
    #[arg(long, value_enum, default_value_t = EventsOut::Stdout)]
    events: EventsOut,
}

fn main() -> Result<()> {
    let args = Args::parse();
    // A video file needs no camera, so no permission prompt (`--list` still does).
    if args.video.is_none() {
        request_camera_permission()?;
    }

    if args.list {
        return list_cameras();
    }

    // Load the model before the camera and the window, so model errors print cleanly
    // and CoreML's slow first compile is over before any latency is measured.
    let model = if args.no_detect {
        None
    } else {
        let model = load_model(&args.model, args.ep)?;
        eprintln!("detect interval: {}", args.detect_interval);
        Some(model)
    };

    // Open the camera or the video file before the window, so permission, format and
    // ffmpeg errors print cleanly.
    let slot = FrameSlot::new();
    let detection = model.map(|model| DetectionSetup {
        model,
        frames: FrameSlot::new(),
        ep: args.ep,
        interval: args.detect_interval,
    });
    let det_frames = detection.as_ref().map(|d| d.frames.clone());
    let (source, info, capture_errors) = open_source(&args, slot.clone(), det_frames)?;
    let events = setup_events(args.rules, args.events, detection.is_some())?;

    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_title("visiotest")
            .with_inner_size([1600.0, 960.0])
            .with_min_inner_size([960.0, 600.0]),
        wgpu_options: eframe::WgpuConfiguration {
            surface: eframe::SurfaceConfig {
                present_mode: if args.no_vsync {
                    wgpu::PresentMode::AutoNoVsync
                } else {
                    wgpu::PresentMode::AutoVsync
                },
                // egui's default is 2 queued frames; keep one for latency.
                desired_maximum_frame_latency: Some(1),
            },
            ..Default::default()
        },
        ..Default::default()
    };
    let matrix = args.matrix;
    let vsync = !args.no_vsync;
    let cam = SourceSetup {
        source,
        slot,
        info,
        errors: capture_errors,
    };
    eframe::run_native(
        "visiotest",
        options,
        Box::new(move |cc| {
            Ok(Box::new(App::new(
                cc, cam, matrix, vsync, detection, events,
            )?))
        }),
    )
    .map_err(|e| anyhow!("{e}"))
}

/// Loads and warms up the model, printing how long each took.
fn load_model(path: &Path, ep: Ep) -> Result<Model> {
    if !path.exists() {
        bail!(
            "model not found: {} — run: cd model-export && uv run export.py",
            path.display()
        );
    }
    let t0 = Instant::now();
    let mut model = Model::load(path, ep)?;
    let load = t0.elapsed();
    let warm_up = model.warm_up().context("warming up the model")?;
    let name = file_name(path);
    // stderr, like every startup line: stdout is reserved for the JSON event lines.
    eprintln!(
        "model: {name} | EP: {} | load {:.1} s | warm-up {} ms",
        ep.label(),
        load.as_secs_f64(),
        warm_up.as_millis(),
    );
    Ok(model)
}

/// Opens the frame source: the video file with `--video`, else the camera. Prints
/// what it opened and returns it with its `SourceInfo` and bad-buffer counter.
fn open_source(
    args: &Args,
    slot: FrameSlot,
    det_frames: Option<FrameSlot>,
) -> Result<(Source, SourceInfo, Arc<AtomicU64>)> {
    match &args.video {
        Some(path) => open_video(path, slot, det_frames),
        None => open_camera(args.camera, slot, det_frames),
    }
}

fn open_camera(
    index: u32,
    slot: FrameSlot,
    det_frames: Option<FrameSlot>,
) -> Result<(Source, SourceInfo, Arc<AtomicU64>)> {
    let (camera, fmt, errors) = capture::camera::open(index, slot, det_frames)?;
    eprintln!(
        "camera: {} | {}x{} @ {} fps, {}",
        camera.info().human_name(),
        fmt.width(),
        fmt.height(),
        fmt.frame_rate(),
        fmt.format(),
    );
    let info = SourceInfo {
        name: camera.info().human_name(),
        format: fmt.format().to_string(),
        width: fmt.width(),
        height: fmt.height(),
        fps: fmt.frame_rate(),
    };
    Ok((Source::Camera(camera), info, errors))
}

fn open_video(
    path: &Path,
    slot: FrameSlot,
    det_frames: Option<FrameSlot>,
) -> Result<(Source, SourceInfo, Arc<AtomicU64>)> {
    let opened = capture::file::open(path, slot, det_frames)?;
    let name = file_name(path);
    let (width, height, fps) = (opened.width, opened.height, opened.fps);
    eprintln!("video: {name} | {width}x{height} @ {fps:.0} fps (looped, decoded by ffmpeg)");
    let info = SourceInfo {
        name,
        format: "YUYV".to_string(),
        width,
        height,
        // At least 1: the status bar's fps check takes 90 % of it, and a slideshow-like
        // file below 0.5 fps would otherwise round to 0.
        fps: (fps.round() as u32).max(1),
    };
    Ok((Source::File(opened.source), info, opened.errors))
}

/// The last part of `path`, for display (`models/yolo11n.onnx` → `yolo11n.onnx`).
fn file_name(path: &Path) -> String {
    path.file_name()
        .unwrap_or(path.as_os_str())
        .to_string_lossy()
        .into_owned()
}

/// Loads the rules file (none if it doesn't exist; a broken one is an error naming the
/// problem) and starts the event sink when events go to stdout. Without a detector
/// there is no engine, so no sink either, but the rules are still loaded for the UI.
fn setup_events(path: PathBuf, out: EventsOut, detection: bool) -> Result<EventsSetup> {
    let rules = match Rules::load(&path)? {
        Some(rules) => {
            eprintln!("rules: {} — {}", path.display(), rules.summary());
            rules
        }
        None => {
            eprintln!(
                "rules: {} not found, starting with none (press E in the window to draw)",
                path.display()
            );
            Rules::default()
        }
    };
    let sink = if detection && out == EventsOut::Stdout {
        eprintln!("events: stdout (JSON lines)");
        Some(EventSink::spawn(std::io::stdout(), WallClock::now())?)
    } else {
        eprintln!("events: off");
        None
    };
    Ok(EventsSetup { rules, path, sink })
}

fn list_cameras() -> Result<()> {
    let cameras = nokhwa::query(ApiBackend::Auto)?;
    if cameras.is_empty() {
        bail!("no cameras found");
    }
    for info in cameras {
        println!(
            "[{}] {} — {}",
            info.index(),
            info.human_name(),
            info.description()
        );
        // `None` makes nokhwa pick 1920x1080@15 YUYV, which the FaceTime HD camera
        // rejects; the format the live run opens with is accepted.
        let requested =
            RequestedFormat::new::<RgbFormat>(RequestedFormatType::AbsoluteHighestFrameRate);
        match Camera::new(info.index().clone(), requested) {
            Ok(mut cam) => {
                let mut formats = cam.compatible_camera_formats()?;
                formats.sort_by_key(|f| {
                    (
                        f.format().to_string(),
                        f.width(),
                        f.height(),
                        f.frame_rate(),
                    )
                });
                for f in &formats {
                    println!(
                        "    {:>5} {}x{} @ {}",
                        f.format(),
                        f.width(),
                        f.height(),
                        f.frame_rate()
                    );
                }
                // nokhwa 0.10.11's AVFoundation backend filters the list with `!=`
                // instead of `==`, so a camera offering only YUYV lists nothing.
                // Show at least the format it opened with.
                if formats.is_empty() {
                    let f = cam.camera_format();
                    println!(
                        "    {:>5} {}x{} @ {}  (opened; nokhwa can't enumerate formats here)",
                        f.format(),
                        f.width(),
                        f.height(),
                        f.frame_rate()
                    );
                }
            }
            Err(e) => println!("    (cannot open: {e})"),
        }
    }
    Ok(())
}

/// On macOS, AVFoundation needs camera permission before any device access.
fn request_camera_permission() -> Result<()> {
    let (tx, rx) = mpsc::channel();
    nokhwa::nokhwa_initialize(move |granted| {
        let _ = tx.send(granted);
    });
    match rx.recv_timeout(Duration::from_secs(30)) {
        Ok(true) => Ok(()),
        Ok(false) => {
            bail!("camera permission denied (System Settings → Privacy & Security → Camera)")
        }
        Err(_) => bail!("timed out waiting for camera permission"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn args_parse_detect_interval() {
        let parse = |extra: &[&str]| {
            let argv = ["visiotest"].iter().chain(extra).copied();
            Args::try_parse_from(argv).map(|a| a.detect_interval)
        };
        assert_eq!(parse(&[]).unwrap(), DetectInterval::EveryFrame);
        assert_eq!(
            parse(&["--detect-interval", "auto"]).unwrap(),
            DetectInterval::Auto
        );
        assert_eq!(
            parse(&["--detect-interval", "250"]).unwrap(),
            DetectInterval::Fixed(Duration::from_millis(250))
        );
        // Above MAX_INTERVAL (500 ms).
        assert!(parse(&["--detect-interval", "900"]).is_err());
    }

    #[test]
    fn args_parse_rules_and_events() {
        let parse = |extra: &[&str]| {
            let argv = ["visiotest"].iter().chain(extra).copied();
            Args::try_parse_from(argv).map(|a| (a.rules, a.events))
        };
        assert_eq!(
            parse(&[]).unwrap(),
            (PathBuf::from("rules.json"), EventsOut::Stdout)
        );
        assert_eq!(
            parse(&["--rules", "x.json", "--events", "off"]).unwrap(),
            (PathBuf::from("x.json"), EventsOut::Off)
        );
        assert!(parse(&["--events", "nats"]).is_err());
    }

    #[test]
    fn args_parse_video() {
        let parse = |extra: &[&str]| {
            let argv = ["visiotest"].iter().chain(extra).copied();
            Args::try_parse_from(argv).map(|a| a.video)
        };
        assert_eq!(parse(&[]).unwrap(), None);
        assert_eq!(
            parse(&["--video", "a.mp4"]).unwrap(),
            Some(PathBuf::from("a.mp4"))
        );
        assert!(parse(&["--video", "a.mp4", "--camera", "1"]).is_err());
    }
}
