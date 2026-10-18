//! YUV → RGB colour maths: the single source of truth for the shader's uniform.

/// YUV → RGB colour matrix, limited (video) range.
#[derive(Clone, Copy, Debug, PartialEq, Eq, clap::ValueEnum)]
pub enum ColorMatrix {
    /// ITU-R BT.709 (HD).
    Bt709,
    /// ITU-R BT.601 (SD; what nokhwa's CPU decoder uses).
    Bt601,
}

/// Uniform for yuyv.wgsl. Layout must match `struct Params` there (48 bytes, all 4-byte scalars).
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct YuvParams {
    pub y_offset: f32,
    pub y_scale: f32,
    pub c_offset: f32,
    pub _pad0: f32,
    pub r_v: f32,
    pub g_u: f32,
    pub g_v: f32,
    pub b_u: f32,
    pub width: u32,
    pub _pad1: u32,
    pub _pad2: u32,
    pub _pad3: u32,
}

impl ColorMatrix {
    /// Shader parameters for a frame `width` pixels wide.
    pub fn params(self, width: u32) -> YuvParams {
        let (kr, kb) = match self {
            ColorMatrix::Bt709 => (0.2126_f32, 0.0722_f32),
            ColorMatrix::Bt601 => (0.299, 0.114),
        };
        let kg = 1.0 - kr - kb;
        let c = 255.0 / 224.0;
        YuvParams {
            y_offset: 16.0 / 255.0,
            y_scale: 255.0 / 219.0,
            c_offset: 128.0 / 255.0,
            _pad0: 0.0,
            r_v: c * 2.0 * (1.0 - kr),
            g_u: c * 2.0 * kb * (1.0 - kb) / kg,
            g_v: c * 2.0 * kr * (1.0 - kr) / kg,
            b_u: c * 2.0 * (1.0 - kb),
            width,
            _pad1: 0,
            _pad2: 0,
            _pad3: 0,
        }
    }
}

/// CPU version of `fs_main` in yuyv.wgsl: one limited-range Y'CbCr sample → R'G'B'.
///
/// Y, U and V are taken as `0..=255` values in `f32`, so callers can pass samples that
/// were already interpolated between pixels (the detector's bilinear resize does). The
/// result is gamma-encoded R'G'B' clamped to `[0, 1]`, like the shader writes.
pub fn yuv_to_rgb_f32(p: &YuvParams, y: f32, u: f32, v: f32) -> [f32; 3] {
    let yy = (y / 255.0 - p.y_offset) * p.y_scale;
    let u = u / 255.0 - p.c_offset;
    let v = v / 255.0 - p.c_offset;
    let rgb = [yy + p.r_v * v, yy - p.g_u * u - p.g_v * v, yy + p.b_u * u];
    rgb.map(|x| x.clamp(0.0, 1.0))
}

/// 8-bit CPU reference that mirrors `fs_main` in yuyv.wgsl exactly.
#[cfg(test)]
pub fn yuv_to_rgb(p: &YuvParams, y: u8, u: u8, v: u8) -> [u8; 3] {
    yuv_to_rgb_f32(p, f32::from(y), f32::from(u), f32::from(v)).map(|x| (x * 255.0).round() as u8)
}

#[cfg(test)]
mod tests {
    use super::*;

    const BOTH: [ColorMatrix; 2] = [ColorMatrix::Bt709, ColorMatrix::Bt601];

    #[test]
    fn params_size_matches_wgsl() {
        assert_eq!(std::mem::size_of::<YuvParams>(), 48);
    }

    #[test]
    fn color_black_white_grey() {
        for m in BOTH {
            let p = m.params(1920);
            assert_eq!(yuv_to_rgb(&p, 16, 128, 128), [0, 0, 0], "{m:?}");
            assert_eq!(yuv_to_rgb(&p, 235, 128, 128), [255, 255, 255], "{m:?}");
            let [r, g, b] = yuv_to_rgb(&p, 126, 128, 128);
            assert!(r == g && g == b, "{m:?}: {r} {g} {b}");
        }
    }

    #[test]
    fn color_f32_black_white() {
        for m in BOTH {
            let p = m.params(1920);
            assert_eq!(
                yuv_to_rgb_f32(&p, 16.0, 128.0, 128.0),
                [0.0, 0.0, 0.0],
                "{m:?}"
            );
            for x in yuv_to_rgb_f32(&p, 235.0, 128.0, 128.0) {
                assert!((x - 1.0).abs() < 1e-5, "{m:?}: {x}");
            }
        }
    }

    #[test]
    fn color_bt601_matches_nokhwa() {
        let p = ColorMatrix::Bt601.params(1920);
        for y in (16..=235).step_by(7) {
            for u in (16..=240).step_by(16) {
                for v in (16..=240).step_by(16) {
                    let ours = yuv_to_rgb(&p, y as u8, u as u8, v as u8);
                    let theirs = nokhwa::utils::yuyv444_to_rgb(y, u, v);
                    for c in 0..3 {
                        let d = (i32::from(ours[c]) - i32::from(theirs[c])).abs();
                        assert!(d <= 2, "yuv=({y},{u},{v}) ours={ours:?} nokhwa={theirs:?}");
                    }
                }
            }
        }
    }
}
