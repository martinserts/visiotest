//! `Latest<T>`: a latest-only, one-value handoff between two threads.
//!
//! Every handoff in the pipeline (camera → UI, camera → detector, detector → UI) is one
//! of these. It never queues more than one value: a new value replaces an unconsumed
//! one, and the replaced value counts as dropped. Latency matters more than
//! completeness, so a slow consumer always gets the newest value and never works
//! through a backlog of stale ones.

use std::sync::{Arc, Condvar, Mutex, MutexGuard};
use std::time::Duration;

struct Inner<T> {
    value: Option<T>,
    dropped: u64,
}

/// A shared, latest-only slot. Cloning it gives another handle to the same slot.
pub struct Latest<T>(Arc<(Mutex<Inner<T>>, Condvar)>);

// Written by hand: `#[derive(Clone)]` would require `T: Clone`, but cloning a handle
// only clones the `Arc`.
impl<T> Clone for Latest<T> {
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}

impl<T> Default for Latest<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> Latest<T> {
    pub fn new() -> Self {
        let inner = Inner {
            value: None,
            dropped: 0,
        };
        Self(Arc::new((Mutex::new(inner), Condvar::new())))
    }

    fn lock(&self) -> MutexGuard<'_, Inner<T>> {
        // A panic while holding the lock can't leave the Option half-swapped.
        self.0.0.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Stores `value`, replacing (and counting as dropped) one nobody took yet.
    pub fn put(&self, value: T) {
        let old = {
            let mut inner = self.lock();
            let old = inner.value.replace(value);
            if old.is_some() {
                inner.dropped += 1;
            }
            old
        };
        // Wake a consumer blocked in `wait_take`.
        self.0.1.notify_one();
        // Release the old value outside the lock (for a frame, that frees its buffer).
        drop(old);
    }

    /// Takes the value if there is one. Never blocks (beyond the short lock).
    pub fn take(&self) -> Option<T> {
        self.lock().value.take()
    }

    /// Waits until a value is present or `timeout` expires, then takes it.
    /// The timeout lets a worker thread check its stop flag regularly.
    pub fn wait_take(&self, timeout: Duration) -> Option<T> {
        let guard = self.lock();
        // `wait_timeout_while` handles spurious wake-ups: it only returns early once
        // the condition (`value.is_none()`) is false.
        let (mut inner, _timed_out) = self
            .0
            .1
            .wait_timeout_while(guard, timeout, |inner| inner.value.is_none())
            .unwrap_or_else(|e| e.into_inner());
        inner.value.take()
    }

    /// How many values were overwritten before anyone took them.
    pub fn dropped(&self) -> u64 {
        self.lock().dropped
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Instant;

    #[test]
    fn slot_keeps_latest_and_counts_drops() {
        let slot = Latest::<u64>::new();
        assert!(slot.take().is_none());
        slot.put(1);
        slot.put(2);
        assert_eq!(slot.take(), Some(2));
        assert_eq!(slot.dropped(), 1);
        assert!(slot.take().is_none());
        slot.put(3);
        assert_eq!(slot.dropped(), 1);
        assert_eq!(slot.take(), Some(3));
    }

    #[test]
    fn slot_wait_take_returns_a_present_value_at_once() {
        let slot = Latest::<u64>::new();
        slot.put(7);
        assert_eq!(slot.wait_take(Duration::from_secs(5)), Some(7));
    }

    #[test]
    fn slot_wait_take_times_out() {
        let slot = Latest::<u64>::new();
        let t0 = Instant::now();
        assert_eq!(slot.wait_take(Duration::from_millis(20)), None);
        assert!(t0.elapsed() >= Duration::from_millis(20));
    }

    #[test]
    fn slot_wait_take_wakes_on_put() {
        let slot = Latest::<u64>::new();
        let producer = slot.clone();
        let t = std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(20));
            producer.put(42);
        });
        // The timeout is far longer than the producer's delay: this must wake on `put`.
        let t0 = Instant::now();
        assert_eq!(slot.wait_take(Duration::from_secs(5)), Some(42));
        assert!(t0.elapsed() < Duration::from_secs(4));
        t.join().unwrap();
    }
}
