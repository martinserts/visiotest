//! Preprocessing: raw YUYV camera frame → the model's 1×3×640×640 input tensor.
//!
//! The model wants a square 640×640 RGB image, as `f32` in `[0, 1]`, stored plane by
//! plane (NCHW: all R values, then all G, then all B). The camera gives a 16:9 YUYV
//! frame. We do the whole conversion in **one pass** over the output pixels:
//!
//! 1. **Letterbox:** scale the frame uniformly so it fits inside 640×640 and centre it.
//!    The leftover bars are filled with grey.
//! 2. **Resize (bilinear):** each output pixel looks up where it falls in the source
//!    frame and blends the 4 nearest source pixels.
//! 3. **Colour:** the blended Y, U and V are converted to R'G'B' with the same formula
//!    as the display shader ([`yuv_to_rgb_f32`]).
//!
//! Everything that depends only on the frame size (where each output row and column
//! samples from) is computed once in [`Preprocessor::new`], so [`Preprocessor::run`]
//! allocates nothing.

use super::{BBox, INPUT_SIZE};
use crate::capture::YuyvLayout;
use crate::color::{ColorMatrix, YuvParams, yuv_to_rgb_f32};

/// Side of the input square, as a `usize` for indexing.
const N: usize = INPUT_SIZE as usize;
/// Values in one colour plane of the input tensor.
const PLANE: usize = N * N;
/// Length of the whole input tensor (3 planes, batch of 1).
pub const TENSOR_LEN: usize = 3 * PLANE;
/// Letterbox bar colour: 114/255 grey is what Ultralytics pads with during training,
/// so the model has learned to treat it as "nothing here".
const PAD: f32 = 114.0 / 255.0;

/// Where the frame sits inside the 640×640 input: scale, then centred padding.
///
/// We letterbox instead of stretching to a square because YOLO was trained on images
/// with their real aspect ratio. Squashing 16:9 into 1:1 would make every object look
/// thinner than anything the model has seen.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Letterbox {
    /// Input pixels per frame pixel (the same on both axes).
    pub scale: f32,
    /// Width of the left bar, in input pixels.
    pub pad_x: f32,
    /// Height of the top bar, in input pixels.
    pub pad_y: f32,
    pub frame_w: u32,
    pub frame_h: u32,
}

impl Letterbox {
    pub fn new(frame_w: u32, frame_h: u32) -> Self {
        let side = INPUT_SIZE as f32;
        // The tighter axis decides: 1920×1080 → min(640/1920, 640/1080) = 1/3.
        let scale = (side / frame_w as f32).min(side / frame_h as f32);
        let (new_w, new_h) = scaled_size(frame_w, frame_h, scale);
        Self {
            scale,
            // Centre the content; floor puts an odd leftover pixel on the right/bottom.
            pad_x: ((side - new_w as f32) / 2.0).floor(),
            pad_y: ((side - new_h as f32) / 2.0).floor(),
            frame_w,
            frame_h,
        }
    }

    /// Size of the scaled frame inside the input square, in input pixels.
    pub fn content_size(&self) -> (u32, u32) {
        scaled_size(self.frame_w, self.frame_h, self.scale)
    }

    /// Maps a box from input space (640×640) back to frame pixels, clamped to the frame.
    pub fn to_frame(self, b: BBox) -> BBox {
        let fx = |x: f32| ((x - self.pad_x) / self.scale).clamp(0.0, self.frame_w as f32);
        let fy = |y: f32| ((y - self.pad_y) / self.scale).clamp(0.0, self.frame_h as f32);
        BBox {
            x0: fx(b.x0),
            y0: fy(b.y0),
            x1: fx(b.x1),
            y1: fy(b.y1),
        }
    }
}

fn scaled_size(w: u32, h: u32, scale: f32) -> (u32, u32) {
    let scaled = |len: u32| (len as f32 * scale).round() as u32;
    (scaled(w), scaled(h))
}

/// One bilinear tap: output pixel `d` blends source pixels `i0` and `i1`,
/// weighting `i1` by `w1` (and `i0` by `1 − w1`).
#[derive(Clone, Copy, Debug, PartialEq)]
struct Tap {
    i0: u32,
    i1: u32,
    w1: f32,
}

/// Builds the taps for `dst_len` output pixels sampling a `src_len`-pixel source.
///
/// Pixel centres line up: output pixel `d` covers `[d, d+1)`, whose centre `d + 0.5`
/// maps to source coordinate `(d + 0.5) / scale`, and source pixel `i` has its centre
/// at `i + 0.5`, hence the final `− 0.5`. This is OpenCV's `INTER_LINEAR` rule, which is
/// what Ultralytics resizes with during training, so the model sees the same blur.
fn taps(dst_len: u32, src_len: u32, scale: f32) -> Vec<Tap> {
    let last = (src_len - 1) as f32;
    (0..dst_len)
        .map(|d| {
            // Near the edges `s` can fall just outside the frame (e.g. −0.33), so clamp
            // before turning it into indices.
            let s = ((d as f32 + 0.5) / scale - 0.5).clamp(0.0, last);
            let i0 = s.floor() as u32;
            Tap {
                i0,
                i1: (i0 + 1).min(src_len - 1),
                w1: s - i0 as f32,
            }
        })
        .collect()
}

/// Turns raw YUYV frames of one fixed layout into model input tensors.
pub struct Preprocessor {
    layout: YuyvLayout,
    params: YuvParams,
    letterbox: Letterbox,
    /// One tap per content column: which source columns it blends.
    cols: Vec<Tap>,
    /// One tap per content row: which source rows it blends.
    rows: Vec<Tap>,
}

impl Preprocessor {
    pub fn new(layout: YuyvLayout, matrix: ColorMatrix) -> Self {
        let letterbox = Letterbox::new(layout.width, layout.height);
        let (new_w, new_h) = letterbox.content_size();
        Self {
            layout,
            params: matrix.params(layout.width),
            letterbox,
            cols: taps(new_w, layout.width, letterbox.scale),
            rows: taps(new_h, layout.height, letterbox.scale),
        }
    }

    pub fn letterbox(&self) -> Letterbox {
        self.letterbox
    }

    /// The frame layout these tables were built for.
    pub fn layout(&self) -> YuyvLayout {
        self.layout
    }

    /// Writes a 3×640×640 RGB NCHW tensor in [0, 1] into `out` (len 3·640·640).
    pub fn run(&self, yuyv: &[u8], out: &mut [f32]) {
        assert_eq!(out.len(), TENSOR_LEN, "input tensor must be 3×640×640");
        let l = self.layout;
        assert!(
            yuyv.len() >= l.stride as usize * l.height as usize,
            "YUYV buffer smaller than its layout"
        );
        self.fill_padding(out);
        let (x_off, y_off) = (self.letterbox.pad_x as usize, self.letterbox.pad_y as usize);
        for (dy, row_tap) in self.rows.iter().enumerate() {
            let top = self.source_row(yuyv, row_tap.i0);
            let bottom = self.source_row(yuyv, row_tap.i1);
            let out_row = (y_off + dy) * N + x_off;
            for (dx, col_tap) in self.cols.iter().enumerate() {
                let [y, u, v] = sample_yuv(top, bottom, row_tap.w1, col_tap);
                write_pixel(out, out_row + dx, yuv_to_rgb_f32(&self.params, y, u, v));
            }
        }
    }

    /// The visible bytes of source row `y` (row padding, if any, is cut off).
    fn source_row<'a>(&self, yuyv: &'a [u8], y: u32) -> &'a [u8] {
        let start = y as usize * self.layout.stride as usize;
        &yuyv[start..start + 2 * self.layout.width as usize]
    }

    /// Paints the letterbox bars (everything outside the content rect) in all 3 planes.
    fn fill_padding(&self, out: &mut [f32]) {
        let (new_w, new_h) = self.letterbox.content_size();
        let (x0, y0) = (self.letterbox.pad_x as usize, self.letterbox.pad_y as usize);
        let (x1, y1) = (x0 + new_w as usize, y0 + new_h as usize);
        for plane in out.chunks_exact_mut(PLANE) {
            for (y, row) in plane.chunks_exact_mut(N).enumerate() {
                if y < y0 || y >= y1 {
                    row.fill(PAD);
                } else {
                    row[..x0].fill(PAD);
                    row[x1..].fill(PAD);
                }
            }
        }
    }
}

/// Y, U and V of source pixel `x` in one YUYV row, as `f32` in `0..=255`.
///
/// Each 4-byte group `Y0 U Y1 V` holds two pixels that share one U and one V
/// (4:2:2 chroma subsampling). So pixel `x` has its own Y at byte `2x`, but takes U and
/// V from its pair `x / 2`: an odd `x` gets the same chroma as `x − 1`.
fn pixel(row: &[u8], x: u32) -> [f32; 3] {
    let x = x as usize;
    let pair = 4 * (x / 2);
    [row[2 * x], row[pair + 1], row[pair + 3]].map(f32::from)
}

/// Bilinear sample: blend horizontally within the `top` and `bottom` rows, then
/// blend those two results vertically with weight `wy` on `bottom`.
fn sample_yuv(top: &[u8], bottom: &[u8], wy: f32, col: &Tap) -> [f32; 3] {
    let upper = lerp(pixel(top, col.i0), pixel(top, col.i1), col.w1);
    let lower = lerp(pixel(bottom, col.i0), pixel(bottom, col.i1), col.w1);
    lerp(upper, lower, wy)
}

/// `a + (b − a)·t` per component: `t = 0` gives `a`, `t = 1` gives `b`.
fn lerp(a: [f32; 3], b: [f32; 3], t: f32) -> [f32; 3] {
    std::array::from_fn(|k| a[k] + (b[k] - a[k]) * t)
}

/// Stores one RGB pixel at offset `i` of each plane (NCHW: R plane, G plane, B plane).
fn write_pixel(out: &mut [f32], i: usize, [r, g, b]: [f32; 3]) {
    out[i] = r;
    out[PLANE + i] = g;
    out[2 * PLANE + i] = b;
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A YUYV frame whose pixel `(x, y)` has luma `luma(x, y)`; `chroma(x)` is the byte
    /// after it (U for even `x`, V for odd `x`). Row padding bytes (beyond `2·w`) are
    /// filled with junk so a test notices if they are read.
    fn frame(
        w: u32,
        h: u32,
        stride: u32,
        luma: impl Fn(u32, u32) -> u8,
        chroma: impl Fn(u32) -> u8,
    ) -> Vec<u8> {
        let mut bytes = vec![0xAA; (stride * h) as usize];
        for y in 0..h {
            for x in 0..w {
                let i = (y * stride + 2 * x) as usize;
                bytes[i] = luma(x, y);
                bytes[i + 1] = chroma(x);
            }
        }
        bytes
    }

    fn run(w: u32, h: u32, stride: u32, bytes: &[u8]) -> Vec<f32> {
        let layout = YuyvLayout::from_len(w, h, bytes.len()).unwrap();
        assert_eq!(layout.stride, stride);
        let mut out = vec![0.0; TENSOR_LEN];
        Preprocessor::new(layout, ColorMatrix::Bt709).run(bytes, &mut out);
        out
    }

    #[test]
    fn preprocess_letterbox_geometry() {
        let hd = Letterbox::new(1920, 1080);
        assert!((hd.scale - 1.0 / 3.0).abs() < 1e-6);
        assert_eq!((hd.pad_x, hd.pad_y), (0.0, 140.0));
        assert_eq!(hd.content_size(), (640, 360));
        let vga = Letterbox::new(640, 480);
        assert_eq!((vga.scale, vga.pad_x, vga.pad_y), (1.0, 0.0, 80.0));
    }

    #[test]
    fn preprocess_to_frame_corners_and_clamp() {
        let lb = Letterbox::new(1920, 1080);
        let content = BBox::new(0.0, 140.0, 640.0, 500.0);
        let b = lb.to_frame(content);
        let near = |a: f32, e: f32| (a - e).abs() < 1e-3;
        assert!(near(b.x0, 0.0) && near(b.y0, 0.0), "{b:?}");
        assert!(near(b.x1, 1920.0) && near(b.y1, 1080.0), "{b:?}");
        let outside = BBox::new(-50.0, 0.0, 700.0, 640.0);
        let full = BBox::new(0.0, 0.0, 1920.0, 1080.0);
        assert_eq!(lb.to_frame(outside), full);
    }

    #[test]
    fn preprocess_pixel_takes_chroma_from_its_pair() {
        let row = [10, 20, 11, 30, 12, 40, 13, 50];
        assert_eq!(pixel(&row, 0), [10.0, 20.0, 30.0]);
        assert_eq!(pixel(&row, 1), [11.0, 20.0, 30.0]);
        assert_eq!(pixel(&row, 3), [13.0, 40.0, 50.0]);
    }

    #[test]
    fn preprocess_taps_stay_in_bounds() {
        for t in taps(640, 1920, 1.0 / 3.0)
            .iter()
            .chain(&taps(640, 64, 10.0))
        {
            assert!(t.i0 <= t.i1 && (0.0..1.0).contains(&t.w1), "{t:?}");
        }
        assert_eq!(taps(640, 64, 10.0)[639].i1, 63);
    }

    #[test]
    fn preprocess_uniform_grey_and_padding() {
        let mut bytes = Vec::with_capacity(1920 * 1080 * 2);
        for _ in 0..1920 * 1080 / 2 {
            bytes.extend_from_slice(&[126, 128, 126, 128]);
        }
        let out = run(1920, 1080, 3840, &bytes);
        let p = ColorMatrix::Bt709.params(1920);
        let grey = yuv_to_rgb_f32(&p, 126.0, 128.0, 128.0);
        for (c, plane) in out.chunks_exact(PLANE).enumerate() {
            for (y, row) in plane.chunks_exact(N).enumerate() {
                for &v in row {
                    if (140..500).contains(&y) {
                        assert!((v - grey[c]).abs() < 1e-5, "plane {c} row {y}: {v}");
                    } else {
                        assert_eq!(v, PAD, "plane {c} row {y}");
                    }
                }
            }
        }
    }

    #[test]
    fn preprocess_black_to_white_edge() {
        let (w, h) = (1920, 1080);
        let bytes = frame(
            w,
            h,
            2 * w,
            |x, _| if x < w / 2 { 16 } else { 235 },
            |_| 128,
        );
        let out = run(w, h, 2 * w, &bytes);
        // The green plane of a content row; with neutral chroma R = G = B.
        let g = &out[PLANE + 320 * N..PLANE + 321 * N];
        assert!(g[0] < 0.05, "left edge {}", g[0]);
        assert!(g[N - 1] > 0.95, "right edge {}", g[N - 1]);
        assert!(g.windows(2).all(|p| p[1] >= p[0] - 0.02), "not monotonic");
    }

    #[test]
    fn preprocess_padded_stride_matches_tight() {
        let (w, h) = (64, 48);
        let luma = |x: u32, y: u32| (16 + x * 3 + y) as u8;
        // Chroma varies per pair so any mix-up between pairs would change the tensor.
        let chroma = |x: u32| 100 + x as u8;
        let tight = run(w, h, 2 * w, &frame(w, h, 2 * w, luma, chroma));
        let padded = run(w, h, 2 * w + 64, &frame(w, h, 2 * w + 64, luma, chroma));
        assert!(tight == padded, "row padding leaked into the tensor");
    }
}
