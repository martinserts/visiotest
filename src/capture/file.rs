//! The video-file source: a file played **as if it were the camera**, in real time and
//! looped.
//!
//! - **A pipe of raw frames.** ffmpeg (an external tool, not a linked library) decodes
//!   the file, scales it and writes raw YUYV frames to its stdout, which we read as a
//!   pipe. So there is no decoder crate, and the GPU path stays YUYV-only: a frame from
//!   a file looks exactly like a frame from the webcam. ffprobe tells us the size and
//!   frame rate first, and we pick the output size ourselves ([`output_size`]), so the
//!   number of bytes per frame is known before the first read.
//! - **We pace, not ffmpeg.** ffmpeg's `-re` would read the input at its native rate,
//!   but the timing would be hidden inside ffmpeg. Instead ffmpeg decodes as fast as it
//!   likes, the pipe's backpressure blocks it while we wait, and our own [`Pacer`]
//!   decides when each frame is handed over, in Rust where it can be read and tested.
//! - **`-noautorotate`.** ffmpeg would rotate a phone clip to upright, but ffprobe
//!   reports the stored (unrotated) size, so our byte count would be wrong. Without the
//!   rotation the probed size stays true; a portrait phone clip shows sideways.
//! - **No sensor timestamp.** A file has no exposure time, so `Frame::sensor_age` is
//!   `None` and the Latency page's total leaves out Sensor → app (`App total*`).
//! - **The loop is a scene cut.** When the file restarts, everyone on screen jumps
//!   back to where they were at the start. The tracker sees that as everyone leaving
//!   and new people arriving: old tracks go lost, new IDs are issued.
//!
//! If ffmpeg dies mid-run, the `video` thread ends quietly: ffmpeg's own error is
//! already on our stderr (its stderr is inherited), and the UI's stall banner shows
//! that frames stopped, as it would for a camera.

use std::ffi::OsString;
use std::io::{ErrorKind, Read};
use std::path::Path;
use std::process::{Child, ChildStdout, Command, Stdio};
use std::sync::Arc;
use std::sync::atomic::AtomicU64;
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use anyhow::{Context, Result, anyhow, bail};
use nokhwa::Buffer;
use nokhwa::utils::{FrameFormat, Resolution};

use super::{FrameSink, FrameSlot, unix_now};

/// The largest frame we decode to: the camera's size, so the latency numbers stay
/// comparable with the webcam's (a 4K frame would also be 16 MB per copy).
const MAX_WIDTH: u32 = 1920;
const MAX_HEIGHT: u32 = 1080;

/// What ffprobe reported about the file's first video stream.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Probe {
    pub width: u32,
    pub height: u32,
    /// Frames per second (`avg_frame_rate`), always `> 0`.
    pub fps: f64,
}

/// Parses `ffprobe … -of csv=p=0` output: `"1281,721,30000/1001\n"`.
pub fn parse_probe(text: &str) -> Result<Probe> {
    let trimmed = text.trim();
    let bad = |why: &str| anyhow!("unexpected ffprobe output {trimmed:?}: {why}");
    let fields: Vec<&str> = trimmed.split(',').collect();
    let [width, height, rate] = fields[..] else {
        return Err(bad("expected width,height,frame rate"));
    };
    let width = parse_size(width).ok_or_else(|| bad("bad width"))?;
    let height = parse_size(height).ok_or_else(|| bad("bad height"))?;
    let fps = parse_rate(rate).ok_or_else(|| bad("unknown or bad frame rate"))?;
    Ok(Probe { width, height, fps })
}

/// A width or height: a whole number above zero.
fn parse_size(field: &str) -> Option<u32> {
    field.parse().ok().filter(|&n| n > 0)
}

/// A frame rate as ffprobe prints it, a fraction `num/den`: `30000/1001` is NTSC's
/// 29.97. ffprobe prints `0/0` when it doesn't know, which is rejected here.
fn parse_rate(field: &str) -> Option<f64> {
    let (num, den) = field.split_once('/')?;
    let num: u64 = num.parse().ok()?;
    let den: u64 = den.parse().ok()?;
    if den == 0 || num == 0 {
        return None;
    }
    Some(num as f64 / den as f64)
}

/// The size ffmpeg scales to: fits inside 1920×1080 (keeping the aspect ratio) and
/// is even in both directions, because YUYV packs pixel pairs and `YuyvLayout` also
/// wants an even height.
///
/// Never upscales: a small file stays small. The maths is in integers (`u64`), so a
/// side that should come out at exactly 1080 can't land at 1079.999… and lose two
/// pixels to the rounding.
pub fn output_size(width: u32, height: u32) -> (u32, u32) {
    let (w, h) = (u64::from(width), u64::from(height));
    let (max_w, max_h) = (u64::from(MAX_WIDTH), u64::from(MAX_HEIGHT));
    // Compare the aspect ratios w/h and max_w/max_h by cross-multiplying: the side
    // that sticks out further decides the scale.
    let (w, h) = if w <= max_w && h <= max_h {
        (w, h)
    } else if w * max_h >= h * max_w {
        (max_w, h * max_w / w)
    } else {
        (w * max_h / h, max_h)
    };
    (even_at_least_2(w), even_at_least_2(h))
}

/// Rounds down to an even number, but never below 2 (the smallest YUYV frame).
fn even_at_least_2(n: u64) -> u32 {
    let even = n - n % 2;
    // `n` is at most 1920 here, so it always fits.
    u32::try_from(even.max(2)).unwrap_or(2)
}

/// ffmpeg's arguments: decode `path` (looped forever), scale to `w`×`h`, and write
/// raw YUYV frames to stdout.
pub fn ffmpeg_args(path: &Path, w: u32, h: u32) -> Vec<OsString> {
    let mut args = Vec::new();
    // Quiet: only real errors reach our stderr, and never read our stdin.
    args.extend(os(&["-hide_banner", "-loglevel", "error", "-nostdin"]));
    // Keep the stored orientation, so ffprobe's size stays the decoded size.
    args.extend(os(&["-noautorotate"]));
    // Play forever, like a camera.
    args.extend(os(&["-stream_loop", "-1", "-i"]));
    args.push(path.as_os_str().to_owned());
    // An exact, even size (see `output_size`), so every frame is `w * h * 2` bytes.
    args.extend(os(&["-vf", &format!("scale={w}:{h}")]));
    // Packed YUYV, the GPU path's only format, so nothing else in visiotest changes.
    args.extend(os(&["-f", "rawvideo", "-pix_fmt", "yuyv422"]));
    // `-` is stdout: our pipe.
    args.extend(os(&["-"]));
    args
}

/// Plain strings as command-line arguments.
fn os<'a>(items: &'a [&str]) -> impl Iterator<Item = OsString> + 'a {
    items.iter().map(OsString::from)
}

/// Real-time pacing: when the next frame is due.
///
/// Frames are due one `period` apart from the first one. A frame that is ready early
/// waits for its slot; the schedule advances by exactly one period each time, so the
/// time doesn't drift (a sleep that ends a little late doesn't push every later frame
/// back). A frame that is ready late goes out at once and the schedule restarts from
/// it: **no catch-up burst**, since a burst would only flood the latest-only slots,
/// which keep the newest frame and drop the rest anyway.
pub struct Pacer {
    period: Duration,
    next: Option<Instant>,
}

impl Pacer {
    /// `fps` must be above zero (`parse_probe` guarantees it).
    pub fn new(fps: f64) -> Self {
        Self {
            period: Duration::from_secs_f64(1.0 / fps),
            next: None,
        }
    }

    /// How long to sleep before handing over a frame that is ready at `now`.
    pub fn wait(&mut self, now: Instant) -> Duration {
        match self.next {
            Some(next) if now < next => {
                self.next = Some(next + self.period);
                next - now
            }
            // The first frame, or a late one: hand it over now, schedule from here.
            _ => {
                self.next = Some(now + self.period);
                Duration::ZERO
            }
        }
    }
}

/// A video file played as a camera: ffmpeg decodes, the `video` thread paces.
pub struct FileSource {
    child: Child,
    thread: Option<JoinHandle<()>>,
}

/// A started [`FileSource`] and what it reported (what `camera::open` returns for a
/// camera: the source, its format and the bad-buffer counter).
pub struct Opened {
    /// Keep it alive: dropping it stops ffmpeg and the `video` thread.
    pub source: FileSource,
    /// The decoded size (see [`output_size`]), which every frame has.
    pub width: u32,
    pub height: u32,
    /// The file's frame rate, as probed.
    pub fps: f64,
    /// Buffers the sink rejected.
    pub errors: Arc<AtomicU64>,
}

/// Probes `path`, starts ffmpeg and the `video` thread, which feeds `ui` and, if
/// given, `detector` through a `FrameSink`.
pub fn open(path: &Path, ui: FrameSlot, detector: Option<FrameSlot>) -> Result<Opened> {
    if !path.exists() {
        bail!("video not found: {}", path.display());
    }
    let probe = probe(path)?;
    let (w, h) = output_size(probe.width, probe.height);
    let mut child = spawn_ffmpeg(path, w, h)?;
    let stdout = child.stdout.take().context("ffmpeg has no stdout pipe")?;
    let errors = Arc::new(AtomicU64::new(0));
    let sink = FrameSink::new(ui, detector, errors.clone());
    let pacer = Pacer::new(probe.fps);
    let spawned = std::thread::Builder::new()
        .name("video".into())
        .spawn(move || play(stdout, (w, h), pacer, sink));
    let thread = match spawned {
        Ok(thread) => thread,
        Err(e) => {
            // Don't leave ffmpeg running with nobody reading its pipe.
            let _ = child.kill();
            let _ = child.wait();
            return Err(e).context("starting the video thread");
        }
    };
    let source = FileSource {
        child,
        thread: Some(thread),
    };
    Ok(Opened {
        source,
        width: w,
        height: h,
        fps: probe.fps,
        errors,
    })
}

/// Runs ffprobe on the first video stream and parses its size and frame rate.
fn probe(path: &Path) -> Result<Probe> {
    let output = Command::new("ffprobe")
        .args(["-v", "error", "-select_streams", "v:0"])
        .args(["-show_entries", "stream=width,height,avg_frame_rate"])
        .args(["-of", "csv=p=0"])
        .arg(path)
        .stdin(Stdio::null())
        .output()
        .map_err(|e| match e.kind() {
            ErrorKind::NotFound => {
                anyhow!("ffprobe not found — install ffmpeg (brew install ffmpeg)")
            }
            _ => anyhow!(e).context("running ffprobe"),
        })?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        bail!("ffprobe failed: {}", stderr.trim());
    }
    parse_probe(&String::from_utf8_lossy(&output.stdout))
}

/// Starts ffmpeg with its stdout as our pipe. Its stderr is inherited, so its error
/// lines land on our stderr; its stdout must never be inherited, since our stdout
/// carries only JSON event lines (raw video would flood `| jq`).
fn spawn_ffmpeg(path: &Path, w: u32, h: u32) -> Result<Child> {
    Command::new("ffmpeg")
        .args(ffmpeg_args(path, w, h))
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .spawn()
        .map_err(|e| match e.kind() {
            ErrorKind::NotFound => {
                anyhow!("ffmpeg not found — install ffmpeg (brew install ffmpeg)")
            }
            _ => anyhow!(e).context("starting ffmpeg"),
        })
}

/// The `video` thread: read one frame, wait until it is due, hand it to the sink.
/// Ends when the pipe does (ffmpeg exited or was killed).
fn play(mut pipe: ChildStdout, (w, h): (u32, u32), mut pacer: Pacer, mut sink: FrameSink) {
    // One read buffer for the whole run: 2 bytes per pixel in YUYV.
    let mut frame = vec![0u8; w as usize * h as usize * 2];
    let resolution = Resolution::new(w, h);
    while pipe.read_exact(&mut frame).is_ok() {
        std::thread::sleep(pacer.wait(Instant::now()));
        // The one copy per frame (~4 MB at 1080p): `Buffer::new` copies the bytes into
        // its own `Bytes`, and nokhwa has no constructor that takes ownership. The
        // camera path copies too, inside nokhwa.
        let buffer = Buffer::new(resolution, &frame, FrameFormat::YUYV);
        sink.on_buffer(buffer, Instant::now(), unix_now());
    }
}

impl Drop for FileSource {
    fn drop(&mut self) {
        // Kill first: the thread is usually blocked in `read_exact` on a pipe that a
        // looping ffmpeg would keep open forever, so joining first would deadlock.
        // The kill closes the pipe, the read fails and the loop ends. The kill's error
        // is ignored: ffmpeg may already have exited.
        let _ = self.child.kill();
        // Reap the process, so it doesn't linger as a zombie.
        let _ = self.child.wait();
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn probe_parses_ntsc_rate() {
        let p = parse_probe("1281,721,30000/1001\n").unwrap();
        assert_eq!((p.width, p.height), (1281, 721));
        assert!((p.fps - 29.97).abs() < 0.001);
    }

    #[test]
    fn probe_parses_whole_rate() {
        let p = parse_probe("1920,1080,25/1").unwrap();
        assert_eq!(
            p,
            Probe {
                width: 1920,
                height: 1080,
                fps: 25.0
            }
        );
    }

    #[test]
    fn probe_rejects_bad_output() {
        let err = parse_probe("1920,1080,0/0").unwrap_err().to_string();
        assert!(err.contains("\"1920,1080,0/0\""), "{err}");
        assert!(parse_probe("1920,1080").is_err());
        assert!(parse_probe("1920,tall,25/1").is_err());
        assert!(parse_probe("0,1080,25/1").is_err());
        assert!(parse_probe("").is_err());
    }

    #[test]
    fn output_size_fits_and_is_even() {
        assert_eq!(output_size(1281, 721), (1280, 720));
        assert_eq!(output_size(3840, 2160), (1920, 1080));
        // Portrait: 1080 · 1080 / 1920 = 607.5 → 607 → 606.
        assert_eq!(output_size(1080, 1920), (606, 1080));
        assert_eq!(output_size(640, 480), (640, 480));
        assert_eq!(output_size(3, 3), (2, 2));
    }

    #[test]
    fn ffmpeg_args_exact() {
        let args = ffmpeg_args(Path::new("my clip.mp4"), 1280, 720);
        let expected = [
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-noautorotate",
            "-stream_loop",
            "-1",
            "-i",
            "my clip.mp4",
            "-vf",
            "scale=1280:720",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "yuyv422",
            "-",
        ];
        // The path with a space stays one argument.
        assert_eq!(args, expected.map(OsString::from));
    }

    const PERIOD: Duration = Duration::from_millis(40);

    #[test]
    fn pacer_first_frame_goes_at_once() {
        let mut pacer = Pacer::new(25.0);
        let t0 = Instant::now();
        assert_eq!(pacer.wait(t0), Duration::ZERO);
        assert_eq!(pacer.next, Some(t0 + PERIOD));
    }

    #[test]
    fn pacer_on_time_waits_for_the_slot() {
        let mut pacer = Pacer::new(25.0);
        let t0 = Instant::now();
        pacer.wait(t0);
        // Ready 10 ms after the first frame: wait the other 30 ms.
        let ready = t0 + Duration::from_millis(10);
        assert_eq!(pacer.wait(ready), Duration::from_millis(30));
        assert_eq!(pacer.next, Some(t0 + 2 * PERIOD));
    }

    #[test]
    fn pacer_jitter_keeps_the_schedule() {
        let mut pacer = Pacer::new(25.0);
        let t0 = Instant::now();
        pacer.wait(t0);
        pacer.wait(t0);
        // The sleep ended 3 ms late, and the next frame was ready at once: it waits
        // 3 ms less, so it still goes out on its slot at t0 + 2 periods.
        let late_wake = t0 + PERIOD + Duration::from_millis(3);
        assert_eq!(pacer.wait(late_wake), PERIOD - Duration::from_millis(3));
        assert_eq!(pacer.next, Some(t0 + 3 * PERIOD));
    }

    #[test]
    fn pacer_late_frame_restarts_without_a_burst() {
        let mut pacer = Pacer::new(25.0);
        let t0 = Instant::now();
        pacer.wait(t0);
        // 100 ms late, past the slot at t0 + 40 ms: go now, next one a period later.
        let late = t0 + Duration::from_millis(140);
        assert_eq!(pacer.wait(late), Duration::ZERO);
        assert_eq!(pacer.next, Some(late + PERIOD));
        // The frame right behind it waits a full period, instead of rushing to catch up.
        assert_eq!(pacer.wait(late), PERIOD);
    }

    #[test]
    fn pacer_does_not_drift() {
        let mut pacer = Pacer::new(25.0);
        let t0 = Instant::now();
        let mut now = t0;
        for _ in 0..25 {
            // Each frame is ready 1 ms after the last went out, and the sleep is exact.
            now += Duration::from_millis(1);
            now += pacer.wait(now);
        }
        // The first frame went out at t0 + 1 ms; the 25th, 24 periods after it.
        assert_eq!(now, t0 + Duration::from_millis(1) + 24 * PERIOD);
        assert_eq!(
            pacer.next,
            Some(t0 + Duration::from_millis(1) + 25 * PERIOD)
        );
    }
}
