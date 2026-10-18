//! Kalman filter: a smooth best guess of where a box is, between and during detections.
//!
//! **The problem.** The detector gives us a box now and then, and every box is a bit
//! noisy: it jitters by a few pixels even when the object stands still. Some frames
//! have no detection at all (the detector skipped them, or the object is hidden behind
//! something). We want a smooth estimate of each box at *any* moment, including the
//! moments we didn't measure.
//!
//! **Two numbers per coordinate.** For one coordinate, say the box centre `cx`, the
//! filter keeps
//! - its **position** (px) and **velocity** (px/s), and
//! - how sure it is of them: the **variance** of each (`var_pos`, `var_vel`, the
//!   square of a standard deviation) and their **covariance** (`cov`), which says how an
//!   error in one goes with an error in the other. If the position turns out further
//!   right than predicted, the velocity was probably larger too; `cov` is how the
//!   filter learns velocity from position measurements alone.
//!
//! **Two steps, repeated.**
//! 1. **Predict** (time passes by `dt`): move by `velocity × dt`. Uncertainty
//!    *grows*, because we are guessing: the velocity may be off, and the object may
//!    have sped up or slowed down.
//! 2. **Update** (a measurement arrives): blend the measured value into the estimate,
//!    weighted by the **Kalman gain** `K = P / (P + R)`, where `P` is our position
//!    variance and `R` the detector's. If we are much surer than the detector, `K` is
//!    near 0 and we barely move; if the detector is much surer, `K` is near 1 and we jump
//!    to the measurement. Uncertainty *shrinks*, because we learned something.
//!
//! **Worked example.** We think `cx = 100 px` with variance 16 (std 4 px). The detector
//! says 110 px, also with variance 16. The gain is `16 / (16 + 16) = ½`, so the new
//! position is `100 + ½ · (110 − 100) = 105 px`, and the new variance is
//! `(1 − ½) · 16 = 8`: two equally good guesses average out and are surer together.
//!
//! **Why four 2×2 filters and not one 8×8.** A box has four coordinates, `cx, cy, w, h`,
//! each with a velocity: 8 numbers. The textbook filter (ByteTrack's) keeps an 8×8
//! covariance. But with a constant-velocity model `cx` only ever mixes with its own
//! velocity, never with `h`: every matrix in the filter is block-diagonal with four 2×2
//! blocks, and it stays that way after every predict and update. So the 8-D filter splits
//! **exactly** into four independent 2-D ones, [`Axis`], with the same results and far
//! simpler maths.
//!
//! **Why `dt` in seconds.** The tracker only runs on detector results, and the detector
//! skips frames when it is busy or paced. Measuring time in seconds (not "frames") means
//! a gap of one frame or of half a second is handled by the same formula.

use crate::detect::BBox;

/// Smallest box side the filter will report, in px, so a long coast can never produce
/// a zero or negative size.
pub const MIN_SIZE: f32 = 1.0;

/// Noise settings, in units of the box's own size so one setting fits near and far objects.
///
/// A 400 px person may move 40 px in a frame; a 40 px cup won't. So every noise is a
/// fraction of the box's current width (for `cx`, `w`) or height (for `cy`, `h`).
/// These start from ByteTrack's values; the `sim_report` benchmark retuned `accel_std`
/// and `meas_std` (see `Default`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct KalmanConfig {
    /// Process noise: how much random acceleration we allow, in box sizes / s².
    pub accel_std: f32,
    /// Detector box noise, as a fraction of the box size.
    pub meas_std: f32,
    /// Start uncertainty of the position, as a fraction of the box size.
    pub init_pos_std: f32,
    /// Start uncertainty of the velocity, in box sizes / s.
    pub init_vel_std: f32,
}

impl Default for KalmanConfig {
    fn default() -> Self {
        Self {
            // ByteTrack's per-frame velocity noise (1/160 of the height per frame at
            // 30 fps) accumulates to about 1.0 box size per second of velocity
            // uncertainty. Per second, the setting doesn't depend on the frame rate.
            //
            // Retuned to 0.3 (and `meas_std` to 0.08) by the `sim_report` benchmark: in the
            // `pole` scene the walker's box shrinks to the sliver still visible beside
            // the pole, and its centre slows to half speed. With ByteTrack's values the
            // filter believed those slivers within 3–4 frames, coasted the hidden
            // stretch at ~310 px/s instead of 480 with a 27 px box, missed the walker
            // on the far side and gave it a new id (50 of 50 noise seeds). The stiffer
            // filter keeps more of the speed it learned in the open: 0 of 50 switch.
            // The price is a slower reaction to real changes of speed. In `stop` (a
            // walker stopping dead, then walking back) the box overshoots for a moment
            // after each change: it keeps its id when detecting every frame, but MOTA
            // falls from 0.98 to 0.74, and at a 100 ms detect interval it now switches.
            accel_std: 0.3,
            // ByteTrack's `std_weight_position = 1/20` of the box height was 0.05; 0.08
            // trusts each box a little less, which the `pole` result above needs too
            // (0.3 with 0.05 still switched on 20 of 50 seeds).
            meas_std: 0.08,
            // ByteTrack starts at 2 × 1/20.
            init_pos_std: 0.10,
            // ByteTrack starts at 10 × 1/160 per frame ≈ 1.9 box sizes / s at 30 fps:
            // a new track has no idea which way it moves.
            init_vel_std: 2.0,
        }
    }
}

/// One coordinate (position + velocity) and its 2×2 covariance.
///
/// The covariance is symmetric, so three numbers hold it:
/// `[[var_pos, cov], [cov, var_vel]]`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Axis {
    /// Position, px.
    pub pos: f32,
    /// Velocity, px/s.
    pub vel: f32,
    /// Variance of the position, px².
    var_pos: f32,
    /// Covariance of position and velocity, px²/s.
    cov: f32,
    /// Variance of the velocity, (px/s)².
    var_vel: f32,
}

impl Axis {
    /// Starts at `pos` with velocity 0, with the given standard deviations and no
    /// covariance (we don't know yet how position and velocity errors relate).
    pub fn new(pos: f32, pos_std: f32, vel_std: f32) -> Self {
        Self {
            pos,
            vel: 0.0,
            var_pos: pos_std * pos_std,
            cov: 0.0,
            var_vel: vel_std * vel_std,
        }
    }

    /// Moves the estimate `dt` seconds forward, allowing random acceleration of
    /// standard deviation `accel_std` (px/s²).
    pub fn predict(&mut self, dt: f32, accel_std: f32) {
        // Two results for the same frame (or an older frame) shouldn't happen.
        debug_assert!(dt >= 0.0, "predict with negative dt {dt}");
        if dt <= 0.0 {
            return;
        }
        // q is the white-noise acceleration intensity, (accel_std)².
        let q = accel_std * accel_std;
        // Constant velocity: the position moves by velocity × time.
        self.pos += self.vel * dt;
        // P ← F·P·Fᵀ + Q with F = [[1, dt], [0, 1]]. The F·P·Fᵀ part spreads the
        // velocity's uncertainty into the position. The dt³/3, dt²/2, dt terms are what
        // integrating random acceleration over dt gives. Each line reads only fields
        // that the lines above it haven't changed yet.
        self.var_pos += dt * (2.0 * self.cov + dt * self.var_vel) + q * dt.powi(3) / 3.0;
        self.cov += dt * self.var_vel + q * dt * dt / 2.0;
        self.var_vel += q * dt;
    }

    /// Blends in a measurement of the position with standard deviation `meas_std` (px).
    pub fn update(&mut self, measured: f32, meas_std: f32) {
        // r is the measurement variance, (meas_std)².
        let r = meas_std * meas_std;
        // Innovation variance: how far off the measurement could be from our guess.
        let s = self.var_pos + r;
        // Kalman gain for the position…
        let k_pos = self.var_pos / s;
        // …and for the velocity, which is corrected through the covariance.
        let k_vel = self.cov / s;
        // Innovation: measured − predicted.
        let y = measured - self.pos;
        self.pos += k_pos * y;
        self.vel += k_vel * y;
        // P ← (I − K·H)·P with H = [1, 0], written out. Every line must use the *old*
        // cov, so var_vel goes first and cov last.
        self.var_vel -= k_vel * self.cov;
        self.var_pos *= 1.0 - k_pos;
        self.cov *= 1.0 - k_pos;
    }

    /// Standard deviation of the position, px.
    #[allow(dead_code)] // only the filter's tests read it today
    pub fn pos_std(&self) -> f32 {
        self.var_pos.sqrt()
    }

    /// Standard deviation of the velocity, px/s.
    #[allow(dead_code)] // only the filter's tests read it today
    pub fn vel_std(&self) -> f32 {
        self.var_vel.sqrt()
    }
}

/// d/dt of `(cx, cy, w, h)` in px/s.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct BoxVelocity {
    pub cx: f32,
    pub cy: f32,
    pub w: f32,
    pub h: f32,
}

/// A box's Kalman filter: four independent [`Axis`] filters for `cx`, `cy`, `w`, `h`.
///
/// The x-axis coordinates (`cx`, `w`) take their noise from the current estimated
/// width, the y-axis ones (`cy`, `h`) from the current estimated height.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BoxFilter {
    cx: Axis,
    cy: Axis,
    w: Axis,
    h: Axis,
    cfg: KalmanConfig,
}

impl BoxFilter {
    /// Starts a filter at box `b`, standing still, with size-relative start uncertainty.
    pub fn new(b: BBox, cfg: KalmanConfig) -> Self {
        let (w, h) = (b.width().max(MIN_SIZE), b.height().max(MIN_SIZE));
        let axis = |value: f32, size: f32| {
            Axis::new(value, cfg.init_pos_std * size, cfg.init_vel_std * size)
        };
        Self {
            cx: axis((b.x0 + b.x1) / 2.0, w),
            cy: axis((b.y0 + b.y1) / 2.0, h),
            w: axis(b.width(), w),
            h: axis(b.height(), h),
            cfg,
        }
    }

    /// Moves the box `dt` seconds forward at its estimated velocity.
    pub fn predict(&mut self, dt: f32) {
        let (w, h) = self.size();
        self.cx.predict(dt, self.cfg.accel_std * w);
        self.w.predict(dt, self.cfg.accel_std * w);
        self.cy.predict(dt, self.cfg.accel_std * h);
        self.h.predict(dt, self.cfg.accel_std * h);
    }

    /// Blends in a detected box.
    pub fn update(&mut self, b: BBox) {
        let (w, h) = self.size();
        self.cx.update((b.x0 + b.x1) / 2.0, self.cfg.meas_std * w);
        self.w.update(b.width(), self.cfg.meas_std * w);
        self.cy.update((b.y0 + b.y1) / 2.0, self.cfg.meas_std * h);
        self.h.update(b.height(), self.cfg.meas_std * h);
    }

    /// The current estimate as a box, with sizes clamped to at least [`MIN_SIZE`].
    pub fn bbox(&self) -> BBox {
        let (w, h) = self.size();
        BBox::from_center(self.cx.pos, self.cy.pos, w, h)
    }

    /// The estimated velocity of every coordinate, px/s.
    pub fn velocity(&self) -> BoxVelocity {
        BoxVelocity {
            cx: self.cx.vel,
            cy: self.cy.vel,
            w: self.w.vel,
            h: self.h.vel,
        }
    }

    /// Stops the size from changing (used while a track is lost).
    ///
    /// ByteTrack does this for lost tracks: without a measurement, a box that was
    /// shrinking would keep shrinking for as long as it is hidden.
    pub fn stop_resizing(&mut self) {
        self.w.vel = 0.0;
        self.h.vel = 0.0;
    }

    /// The estimated width and height, clamped to at least [`MIN_SIZE`] so noise that
    /// scales with them never reaches zero.
    fn size(&self) -> (f32, f32) {
        (self.w.pos.max(MIN_SIZE), self.h.pos.max(MIN_SIZE))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f32, b: f32, tol: f32) -> bool {
        (a - b).abs() <= tol
    }

    #[test]
    fn kalman_axis_update_matches_the_worked_example() {
        let mut a = Axis::new(100.0, 4.0, 1.0);
        a.update(110.0, 4.0);
        assert!(close(a.pos, 105.0, 1e-4), "pos {}", a.pos);
        assert!(
            close(a.pos_std(), 8.0_f32.sqrt(), 1e-4),
            "std {}",
            a.pos_std()
        );
    }

    #[test]
    fn kalman_axis_predict_moves_and_grows_uncertainty() {
        let mut a = Axis::new(50.0, 2.0, 3.0);
        a.vel = 10.0;
        let before = a.pos_std();
        a.predict(0.5, 1.0);
        assert!(close(a.pos, 55.0, 1e-4), "pos {}", a.pos);
        assert!(a.pos_std() > before);

        // Without process noise and with a perfectly known velocity, the position
        // is exactly as uncertain after the move as before.
        let mut sure = Axis::new(50.0, 2.0, 0.0);
        sure.vel = 10.0;
        sure.predict(0.5, 0.0);
        assert!(close(sure.pos_std(), 2.0, 1e-6), "std {}", sure.pos_std());
    }

    #[test]
    fn kalman_axis_learns_velocity() {
        let dt = 1.0 / 30.0;
        let mut a = Axis::new(0.0, 2.0, 50.0);
        for step in 1..=60 {
            a.predict(dt, 5.0);
            a.update(30.0 * dt * step as f32, 1.0);
        }
        let truth = 30.0 * 2.0;
        assert!(close(a.vel, 30.0, 1.5), "vel {}", a.vel);
        assert!(close(a.pos, truth, 1.0), "pos {} vs {truth}", a.pos);
    }

    #[test]
    fn kalman_box_round_trips() {
        let b = BBox::new(10.0, 20.0, 110.0, 220.0);
        let out = BoxFilter::new(b, KalmanConfig::default()).bbox();
        for (got, want) in [
            (out.x0, b.x0),
            (out.y0, b.y0),
            (out.x1, b.x1),
            (out.y1, b.y1),
        ] {
            assert!(close(got, want, 1e-4), "{out:?} vs {b:?}");
        }
    }

    #[test]
    fn kalman_box_coasts_through_a_gap() {
        // A 100×200 box moving right at 90 px/s, seen at 30 Hz for 1 s.
        let dt = 1.0 / 30.0;
        let at = |t: f32| BBox::from_center(200.0 + 90.0 * t, 300.0, 100.0, 200.0);
        let mut f = BoxFilter::new(at(0.0), KalmanConfig::default());
        for step in 1..=30 {
            f.predict(dt);
            f.update(at(step as f32 * dt));
        }
        // Then hidden for 0.5 s: no updates, only predictions.
        for _ in 0..15 {
            f.predict(dt);
        }
        let cx = (f.bbox().x0 + f.bbox().x1) / 2.0;
        let truth = 200.0 + 90.0 * 1.5;
        assert!(close(cx, truth, 10.0), "cx {cx} vs {truth}");
    }

    #[test]
    fn kalman_box_never_goes_negative() {
        // A box shrinking by 30 px/s…
        let dt = 1.0 / 30.0;
        let at = |t: f32| BBox::from_center(100.0, 100.0, 50.0 - 30.0 * t, 60.0 - 30.0 * t);
        let mut f = BoxFilter::new(at(0.0), KalmanConfig::default());
        for step in 1..=30 {
            f.predict(dt);
            f.update(at(step as f32 * dt));
        }
        assert!(f.velocity().w < 0.0 && f.velocity().h < 0.0);
        // …then coasting for 10 s, long enough to shrink through zero.
        f.predict(10.0);
        let b = f.bbox();
        assert!(b.width() >= MIN_SIZE && b.height() >= MIN_SIZE, "{b:?}");

        f.stop_resizing();
        assert_eq!(f.velocity().w, 0.0);
        assert_eq!(f.velocity().h, 0.0);
    }

    #[test]
    fn kalman_predict_zero_dt_is_noop() {
        let mut a = Axis::new(7.0, 2.0, 3.0);
        a.vel = 4.0;
        let before = a;
        a.predict(0.0, 1.0);
        assert_eq!(a, before);

        let mut f = BoxFilter::new(BBox::new(0.0, 0.0, 10.0, 20.0), KalmanConfig::default());
        let before = f;
        f.predict(0.0);
        assert_eq!(f, before);
    }
}
