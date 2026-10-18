//! Frame capture: **two sources, one sink**.
//!
//! A frame can come from the webcam ([`camera`], nokhwa's capture thread) or from a
//! video file ([`file`], ffmpeg decoding on a pipe, read by our `video` thread). Both
//! end the same way: whatever produced the raw YUYV bytes wraps them in a
//! `nokhwa::Buffer` and calls [`FrameSink::on_buffer`]. The sink stamps the `Frame`
//! (`seq`, arrival time, interval, fps) and fans it out to latest-only slots: one for
//! the UI and, when detection is on, one for the detector worker. Nothing after the
//! sink knows which source the frame came from.
//!
//! The sink never decodes or copies pixels: the raw YUYV `Buffer` is moved into the
//! slots, and the UI uploads it to the GPU as-is.

pub mod camera;
pub mod file;

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use anyhow::Result;
use nokhwa::{Buffer, CallbackCamera};

use crate::slot::Latest;
use crate::stats::FpsCounter;

/// What produces frames: the webcam or a video file.
///
/// Kept alive by `App`; dropping it stops the frames (nokhwa closes the stream;
/// `FileSource` kills ffmpeg and joins its thread).
#[allow(dead_code)] // the fields are never read, only held for what their `Drop` does
pub enum Source {
    Camera(CallbackCamera),
    File(file::FileSource),
}

/// Geometry of a packed YUYV frame (2 bytes per pixel, `Y0 U Y1 V` per pixel pair).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct YuyvLayout {
    pub width: u32,
    pub height: u32,
    /// Bytes per row, including any row padding the driver added.
    pub stride: u32,
}

impl YuyvLayout {
    /// Derives the row stride from the buffer length (which may include row padding).
    pub fn from_len(width: u32, height: u32, len: usize) -> Result<Self> {
        let bad = || anyhow::anyhow!("bad YUYV buffer: {width}x{height}, {len} bytes");
        if width == 0 || height == 0 || !width.is_multiple_of(2) || !height.is_multiple_of(2) {
            return Err(bad());
        }
        let tight = width as usize * 2;
        let h = height as usize;
        let stride = if len == tight * h {
            tight
        } else if len.is_multiple_of(h) && len / h >= tight && (len / h).is_multiple_of(4) {
            len / h
        } else {
            return Err(bad());
        };
        let stride = u32::try_from(stride).map_err(|_| bad())?;
        Ok(Self {
            width,
            height,
            stride,
        })
    }
}

/// One camera frame, as delivered by the driver.
///
/// `Clone` is cheap: `Buffer` holds its bytes in a reference-counted `bytes::Bytes`, so
/// a clone shares the pixels instead of copying them.
#[derive(Clone)]
pub struct Frame {
    /// Raw YUYV bytes, as delivered (no copy).
    pub buffer: Buffer,
    pub layout: YuyvLayout,
    /// 1-based camera frame counter.
    pub seq: u64,
    /// When the callback ran (the frame reached user space).
    pub received: Instant,
    /// `received` − sensor timestamp, if nokhwa had one.
    pub sensor_age: Option<Duration>,
    /// `received` − previous frame's `received`.
    pub interval: Option<Duration>,
    /// FpsCounter over `received`.
    pub camera_fps: f64,
}

/// Latest-only handoff of camera frames (see [`Latest`]). The UI has one, and the
/// detector has its own, so a busy detector never holds frames back from the video.
pub type FrameSlot = Latest<Frame>;

/// Turns raw nokhwa buffers into `Frame`s. Pure apart from the slots.
struct FrameSink {
    ui: FrameSlot,
    /// `None` with `--no-detect`.
    detector: Option<FrameSlot>,
    fps: FpsCounter,
    prev: Option<Instant>,
    seq: u64,
    errors: Arc<AtomicU64>,
}

impl FrameSink {
    fn new(ui: FrameSlot, detector: Option<FrameSlot>, errors: Arc<AtomicU64>) -> Self {
        Self {
            ui,
            detector,
            fps: FpsCounter::new(),
            prev: None,
            seq: 0,
            errors,
        }
    }

    /// `now` is the monotonic arrival time, `wall_now` the same moment as Unix time.
    fn on_buffer(&mut self, buffer: Buffer, now: Instant, wall_now: Duration) {
        let res = buffer.resolution();
        let layout = match YuyvLayout::from_len(res.width(), res.height(), buffer.buffer().len()) {
            Ok(l) => l,
            Err(_) => {
                self.errors.fetch_add(1, Ordering::Relaxed);
                return;
            }
        };
        self.seq += 1;
        let interval = self.prev.map(|p| now - p);
        self.prev = Some(now);
        let sensor_age = buffer
            .capture_timestamp()
            .map(|ts| wall_now.saturating_sub(ts));
        let camera_fps = self.fps.tick(now);
        let frame = Frame {
            buffer,
            layout,
            seq: self.seq,
            received: now,
            sensor_age,
            interval,
            camera_fps,
        };
        // Fan out. The clone shares the pixel bytes (a refcount bump, not a copy), so
        // both consumers see the same frame at the cost of one atomic increment.
        if let Some(d) = &self.detector {
            d.put(frame.clone());
        }
        self.ui.put(frame);
    }
}

/// Now as Unix time, the clock nokhwa's sensor timestamps use (`wall_now` in
/// [`FrameSink::on_buffer`]). Before 1970 (a broken clock) reads as zero.
fn unix_now() -> Duration {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;
    use nokhwa::utils::{FrameFormat, Resolution};

    #[test]
    fn layout_tight_1080p() {
        let l = YuyvLayout::from_len(1920, 1080, 4_147_200).unwrap();
        assert_eq!(l.stride, 3840);
    }

    #[test]
    fn layout_padded_rows() {
        let l = YuyvLayout::from_len(1920, 1080, 3904 * 1080).unwrap();
        assert_eq!(l.stride, 3904);
    }

    #[test]
    fn layout_rejects_bad_input() {
        assert!(YuyvLayout::from_len(1921, 1080, 1921 * 2 * 1080).is_err());
        assert!(YuyvLayout::from_len(1920, 1080, 4_147_200 - 2).is_err());
        assert!(YuyvLayout::from_len(1920, 1080, 3839 * 1080).is_err());
        assert!(YuyvLayout::from_len(1920, 0, 0).is_err());
    }

    #[test]
    fn sink_fills_frame_fields() {
        let slot = FrameSlot::new();
        let errors = Arc::new(AtomicU64::new(0));
        let mut sink = FrameSink::new(slot.clone(), None, errors.clone());
        let t0 = Instant::now();
        let wall0 = Duration::from_secs(1_000_000);
        let res = Resolution::new(4, 2);

        sink.on_buffer(Buffer::new(res, &[0; 16], FrameFormat::YUYV), t0, wall0);
        let f = slot.take().unwrap();
        assert_eq!(f.seq, 1);
        assert_eq!(f.interval, None);
        assert_eq!(f.sensor_age, None);
        assert_eq!(f.layout.stride, 8);

        let ts = wall0 + Duration::from_millis(10);
        let b = Buffer::with_timestamp(res, &[0; 16], FrameFormat::YUYV, Some(ts));
        let t1 = t0 + Duration::from_millis(33);
        let wall1 = wall0 + Duration::from_millis(33);
        sink.on_buffer(b, t1, wall1);
        let f = slot.take().unwrap();
        assert_eq!(f.seq, 2);
        assert_eq!(f.interval, Some(Duration::from_millis(33)));
        assert_eq!(f.sensor_age, Some(Duration::from_millis(23)));
        assert_eq!(f.received, t1);
        assert_eq!(errors.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn sink_counts_bad_buffers() {
        let slot = FrameSlot::new();
        let errors = Arc::new(AtomicU64::new(0));
        let mut sink = FrameSink::new(slot.clone(), None, errors.clone());
        let b = Buffer::new(Resolution::new(4, 2), &[0; 15], FrameFormat::YUYV);
        sink.on_buffer(b, Instant::now(), Duration::ZERO);
        assert_eq!(errors.load(Ordering::Relaxed), 1);
        assert!(slot.take().is_none());
    }

    #[test]
    fn sink_fans_out_to_detector() {
        let ui = FrameSlot::new();
        let det = FrameSlot::new();
        let errors = Arc::new(AtomicU64::new(0));
        let mut sink = FrameSink::new(ui.clone(), Some(det.clone()), errors);
        let res = Resolution::new(4, 2);
        let pixels: Vec<u8> = (0..16).collect();
        sink.on_buffer(
            Buffer::new(res, &pixels, FrameFormat::YUYV),
            Instant::now(),
            Duration::ZERO,
        );

        let (a, b) = (ui.take().unwrap(), det.take().unwrap());
        assert_eq!((a.seq, b.seq), (1, 1));
        // Same pixels, and the same memory: the clone shared the bytes.
        assert_eq!(a.buffer.buffer(), &pixels[..]);
        assert_eq!(a.buffer.buffer().as_ptr(), b.buffer.buffer().as_ptr());
    }
}
