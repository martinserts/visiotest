//! Small timing helpers shared by the capture thread and the UI.

use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

pub fn ms(d: Duration) -> f64 {
    d.as_secs_f64() * 1000.0
}

/// Seconds-within-the-minute with millisecond precision, e.g. "42.137".
pub fn wall_clock_ms() -> String {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    format!("{:02}.{:03}", now.as_secs() % 60, now.subsec_millis())
}

/// FPS averaged over back-to-back one-second windows: `tick` keeps returning the last
/// finished window's rate until the next one closes.
pub struct FpsCounter {
    window_start: Instant,
    frames: u32,
    fps: f64,
}

impl FpsCounter {
    pub fn new() -> Self {
        Self {
            window_start: Instant::now(),
            frames: 0,
            fps: 0.0,
        }
    }

    pub fn tick(&mut self, now: Instant) -> f64 {
        self.frames += 1;
        let elapsed = now - self.window_start;
        if elapsed >= Duration::from_secs(1) {
            self.fps = f64::from(self.frames) / elapsed.as_secs_f64();
            self.frames = 0;
            self.window_start = now;
        }
        self.fps
    }
}

/// Window length of [`RollingStats`]: ~8 s at 30 fps.
pub const CAP: usize = 240;

/// `last`, `p50` and `p95` of a [`RollingStats`] window, in ms.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Summary {
    pub last: f32,
    pub p50: f32,
    pub p95: f32,
}

/// A fixed-capacity ring of ms values with nearest-rank percentiles.
/// Both buffers are allocated in `new()`; `push` and `percentile` never allocate.
pub struct RollingStats {
    ring: Vec<f32>,
    scratch: Vec<f32>,
    /// Index the next push writes to (once the ring is full).
    head: usize,
}

impl RollingStats {
    pub fn new() -> Self {
        Self {
            ring: Vec::with_capacity(CAP),
            scratch: Vec::with_capacity(CAP),
            head: 0,
        }
    }

    pub fn push(&mut self, v: f32) {
        if self.ring.len() < CAP {
            self.ring.push(v);
        } else {
            self.ring[self.head] = v;
        }
        self.head = (self.head + 1) % CAP;
    }

    pub fn len(&self) -> usize {
        self.ring.len()
    }

    pub fn is_empty(&self) -> bool {
        self.ring.is_empty()
    }

    /// The newest value.
    pub fn last(&self) -> Option<f32> {
        if self.is_empty() {
            return None;
        }
        Some(self.ring[(self.head + CAP - 1) % CAP])
    }

    /// Nearest-rank percentile, `q` in `0.0..=1.0`.
    pub fn percentile(&mut self, q: f32) -> Option<f32> {
        let n = self.len();
        if n == 0 {
            return None;
        }
        self.scratch.clear();
        self.scratch.extend_from_slice(&self.ring);
        let rank = ((q.clamp(0.0, 1.0) * n as f32).ceil() as usize).clamp(1, n);
        let (_, v, _) = self
            .scratch
            .select_nth_unstable_by(rank - 1, f32::total_cmp);
        Some(*v)
    }

    pub fn summary(&mut self) -> Option<Summary> {
        Some(Summary {
            last: self.last()?,
            p50: self.percentile(0.50)?,
            p95: self.percentile(0.95)?,
        })
    }
}

/// True on the first call, then at most once per `period`.
pub struct Throttle {
    pub period: Duration,
    pub next: Option<Instant>,
}

impl Throttle {
    pub fn new(period: Duration) -> Self {
        Self { period, next: None }
    }

    pub fn due(&mut self, now: Instant) -> bool {
        if self.next.is_some_and(|next| now < next) {
            return false;
        }
        self.next = Some(now + self.period);
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A counter plus a start instant at (or just after) its window start.
    fn counter() -> (FpsCounter, Instant) {
        let c = FpsCounter::new();
        let start = c.window_start;
        (c, start)
    }

    #[test]
    fn fps_zero_before_first_second() {
        let (mut c, start) = counter();
        for i in 1..=20 {
            assert_eq!(c.tick(start + Duration::from_millis(i * 40)), 0.0);
        }
    }

    #[test]
    fn fps_thirty_ticks_over_one_second() {
        let (mut c, start) = counter();
        let mut fps = 0.0;
        for i in 1..=30u32 {
            fps = c.tick(start + Duration::from_secs(1) * i / 30);
        }
        assert!((fps - 30.0).abs() < 0.01, "fps = {fps}");
    }

    #[test]
    fn fps_window_resets_after_report() {
        let (mut c, start) = counter();
        for i in 1..=30u32 {
            c.tick(start + Duration::from_secs(1) * i / 30);
        }
        // Second window: 10 ticks over the next second → 10 fps.
        let second = start + Duration::from_secs(1);
        let mut fps = 0.0;
        for i in 1..=10u32 {
            fps = c.tick(second + Duration::from_secs(1) * i / 10);
            if i < 10 {
                assert!((fps - 30.0).abs() < 0.01, "kept the last report until full");
            }
        }
        assert!((fps - 10.0).abs() < 0.01, "fps = {fps}");
    }

    #[test]
    fn rolling_empty_is_none() {
        let mut r = RollingStats::new();
        assert_eq!(r.len(), 0);
        assert_eq!(r.last(), None);
        assert_eq!(r.percentile(0.5), None);
        assert_eq!(r.summary(), None);
    }

    #[test]
    fn rolling_nearest_rank_percentiles() {
        let mut r = RollingStats::new();
        for v in 1..=100 {
            r.push(v as f32);
        }
        let s = r.summary().unwrap();
        assert_eq!((s.last, s.p50, s.p95), (100.0, 50.0, 95.0));
    }

    #[test]
    fn rolling_evicts_oldest() {
        let mut r = RollingStats::new();
        for v in 1..=(CAP + 10) {
            r.push(v as f32);
        }
        assert_eq!(r.len(), CAP);
        assert_eq!(r.percentile(0.0), Some(11.0));
        assert_eq!(r.last(), Some((CAP + 10) as f32));
    }

    #[test]
    fn throttle_due_once_per_period() {
        let mut t = Throttle::new(Duration::from_millis(250));
        let t0 = Instant::now();
        let at = |ms| t0 + Duration::from_millis(ms);
        assert!(t.due(t0));
        assert!(!t.due(at(100)));
        assert!(t.due(at(250)));
        // Next due time counts from the last due call (250), not the first.
        assert!(!t.due(at(400)));
        assert!(t.due(at(500)));
    }
}
