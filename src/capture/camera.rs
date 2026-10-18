//! The webcam source: nokhwa's `CallbackCamera` captures on its own thread and calls
//! our callback once per frame. The callback does no pixel work: it hands the raw
//! YUYV `Buffer` to the [`FrameSink`], which fans it out to the slots.
//!
//! nokhwa's thread loop silently ignores `frame()` errors, so a camera that stops
//! delivering shows up only as frames getting old; the UI flags that as a stall.

use std::sync::Arc;
use std::sync::atomic::AtomicU64;
use std::time::Instant;

use anyhow::{Context, Result, bail};
use nokhwa::pixel_format::RgbFormat;
use nokhwa::utils::{CameraFormat, CameraIndex, FrameFormat, RequestedFormat, RequestedFormatType};
use nokhwa::{Buffer, CallbackCamera};

use super::{FrameSink, FrameSlot, unix_now};

/// Opens the camera and starts nokhwa's capture thread, which feeds `ui` and, if given,
/// `detector`. Returns the camera (keep it alive: dropping it stops the stream), its
/// format, and a counter of buffers that could not be interpreted as YUYV.
pub fn open(
    index: u32,
    ui: FrameSlot,
    detector: Option<FrameSlot>,
) -> Result<(CallbackCamera, CameraFormat, Arc<AtomicU64>)> {
    let errors = Arc::new(AtomicU64::new(0));
    let mut sink = FrameSink::new(ui, detector, errors.clone());
    let callback = move |b: Buffer| sink.on_buffer(b, Instant::now(), unix_now());
    let requested =
        RequestedFormat::new::<RgbFormat>(RequestedFormatType::AbsoluteHighestFrameRate);
    let mut camera = CallbackCamera::new(CameraIndex::Index(index), requested, callback)
        .context("opening camera")?;
    camera.open_stream().context("starting stream")?;
    let fmt = camera.camera_format().context("reading camera format")?;
    if fmt.format() != FrameFormat::YUYV {
        bail!(
            "camera delivers {}, only YUYV is supported by the GPU path",
            fmt.format()
        );
    }
    Ok((camera, fmt, errors))
}
