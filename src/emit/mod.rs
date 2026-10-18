//! Emit: get events out of the process, one JSON line each.
//!
//! The detector thread produces events ([`crate::events::event::Event`]); this module
//! prints them on its own thread, `"events"`, so a slow terminal or pipe never slows
//! the detector down:
//!
//! ```text
//! detector thread ──EventSender::send──► [ bounded queue, QUEUE slots ] ──► "events" thread ──► stdout
//!                   (try_send, never blocks)                               (one JSON line per event)
//! ```
//!
//! **Why not a latest-value slot, like every other handoff?** For frames and results
//! only the newest one matters, so overwriting an old one is the point. An event is
//! different: an overwritten `line_cross` is a person who was never counted. So events
//! go through a *queue*, and every one of them is printed in order.
//!
//! **Why is the queue bounded anyway?** The consumer can get stuck: a terminal paused
//! with Ctrl-S, a pipe into a program that stopped reading. An unbounded queue would
//! then grow forever, and a blocking one would stall the detector, and with it the
//! tracker. Neither is acceptable, so the queue holds [`QUEUE`] events, and
//! [`EventSender::send`] uses `try_send`.
//!
//! **What happens when it is full?** The event is dropped and **counted**
//! ([`EventSink::dropped`]), so the loss is visible rather than silent. The counts in
//! the UI stay exact even then, because they come from the engine's running totals,
//! which travel with each detector result, not from this stream.
//!
//! Stdout is the only consumer. Another one, e.g. publishing to NATS, would sit beside
//! it in this module, without touching the engine.

pub mod json;

pub use json::{WallClock, event_line};

use std::io::Write;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{Receiver, SyncSender, sync_channel};
use std::thread::{self, JoinHandle};

use anyhow::{Context, Result};

use crate::events::event::Event;

/// Where the event lines go (`--events`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, clap::ValueEnum)]
pub enum EventsOut {
    /// JSON lines on stdout.
    Stdout,
    /// No event output (counts still show in the UI).
    Off,
}

/// How many events the queue holds. A busy doorway makes a few events per second, so
/// this is roughly 30 s or more of them: plenty of headroom for a slow terminal, and
/// still tiny in memory.
pub const QUEUE: usize = 1024;

/// Cloneable; never blocks: a full queue drops the event and counts it.
#[derive(Clone)]
pub struct EventSender {
    tx: SyncSender<Event>,
    dropped: Arc<AtomicU64>,
}

impl EventSender {
    /// Queues `e` for printing. If the queue is full (or the sink is gone), the event is
    /// dropped and counted instead; this never waits.
    pub fn send(&self, e: Event) {
        if self.tx.try_send(e).is_err() {
            self.dropped.fetch_add(1, Ordering::Relaxed);
        }
    }
}

/// The "events" thread and its bounded queue. Dropping it flushes and joins.
pub struct EventSink {
    /// The sink's own sender, which `sender()` clones. `None` only while dropping.
    tx: Option<SyncSender<Event>>,
    thread: Option<JoinHandle<()>>,
    /// Events lost to a full queue or a failed write.
    dropped: Arc<AtomicU64>,
}

impl EventSink {
    /// Starts the `"events"` thread, which writes one JSON line per event to `out`,
    /// with `t` mapped to Unix time by `clock`.
    pub fn spawn(out: impl Write + Send + 'static, clock: WallClock) -> Result<EventSink> {
        let (tx, rx) = sync_channel(QUEUE);
        let dropped = Arc::new(AtomicU64::new(0));
        let thread_dropped = dropped.clone();
        let thread = thread::Builder::new()
            .name("events".into())
            .spawn(move || write_loop(rx, out, clock, &thread_dropped))
            .context("starting the events thread")?;
        Ok(EventSink {
            tx: Some(tx),
            thread: Some(thread),
            dropped,
        })
    }

    /// A new sender into this sink's queue, e.g. for the detector thread.
    pub fn sender(&self) -> EventSender {
        let tx = self
            .tx
            .clone()
            .expect("the sink's sender is only taken while dropping");
        sender_for(tx, self.dropped.clone())
    }

    /// The number of events dropped so far (full queue or failed write), shared with
    /// the thread, for the UI.
    pub fn dropped(&self) -> Arc<AtomicU64> {
        self.dropped.clone()
    }
}

impl Drop for EventSink {
    fn drop(&mut self) {
        // The thread's `for e in rx` ends only when *every* sender is gone, this one
        // included, so drop ours first. Any other `EventSender` (the detector holds one)
        // must already be dropped, or this join waits for it: the `Detector` owns this
        // sink and joins its thread before dropping it. After that, the thread only has
        // to write what is still queued.
        self.tx.take();
        if let Some(t) = self.thread.take() {
            let _ = t.join();
        }
    }
}

/// Builds a sender that counts its drops in `dropped`. Separate from `sender()` so the
/// tests can make one without a thread.
fn sender_for(tx: SyncSender<Event>, dropped: Arc<AtomicU64>) -> EventSender {
    EventSender { tx, dropped }
}

/// The thread's loop: write each event as one line, and flush whenever the queue has
/// run empty. Ends when every sender has been dropped.
fn write_loop(rx: Receiver<Event>, out: impl Write, clock: WallClock, dropped: &AtomicU64) {
    let mut lines = LineWriter {
        out,
        clock,
        broken: false,
    };
    // `recv` blocks until an event arrives, and fails once every sender is gone.
    while let Ok(first) = rx.recv() {
        lines.write(&first, dropped);
        // Write the rest of a burst before flushing: lines appear promptly, and a
        // burst costs one flush rather than one per event.
        while let Ok(next) = rx.try_recv() {
            lines.write(&next, dropped);
        }
        lines.flush();
    }
}

/// Writes event lines to `out` until the first error, then only counts.
struct LineWriter<W: Write> {
    out: W,
    clock: WallClock,
    /// Set by the first write or flush error (e.g. stdout piped into `head`, which has
    /// exited). From then on nothing is written, but the queue is still drained so
    /// senders never find it full forever.
    broken: bool,
}

impl<W: Write> LineWriter<W> {
    /// Writes `e` as one line, or counts it in `dropped` if the output is broken.
    fn write(&mut self, e: &Event, dropped: &AtomicU64) {
        if !self.broken {
            let mut line = event_line(e, &self.clock);
            line.push('\n');
            // One `write_all` per line (not `println!`, which panics on a closed
            // stdout): the error comes back here instead.
            match self.out.write_all(line.as_bytes()) {
                Ok(()) => return,
                Err(err) => self.break_with(&err),
            }
        }
        dropped.fetch_add(1, Ordering::Relaxed);
    }

    /// Pushes buffered lines out. A piped stdout is block-buffered, so without this the
    /// lines would sit in the buffer. (Lines lost by a failing flush aren't counted:
    /// we can't tell how many of them had already gone out.)
    fn flush(&mut self) {
        if self.broken {
            return;
        }
        if let Err(err) = self.out.flush() {
            self.break_with(&err);
        }
    }

    /// Reports the first error once and stops writing.
    fn break_with(&mut self, err: &std::io::Error) {
        eprintln!("events: {err}");
        self.broken = true;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::events::event::{Direction, EventKind};
    use std::io;
    use std::sync::Mutex;
    use std::time::Instant;

    /// A writer whose bytes the test can read after the sink has finished.
    struct Shared(Arc<Mutex<Vec<u8>>>);

    impl Write for Shared {
        fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
            self.0.lock().unwrap().extend_from_slice(buf);
            Ok(buf.len())
        }

        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }

    /// A writer that fails like stdout piped into `head` after `head` has exited.
    struct BrokenPipe;

    impl Write for BrokenPipe {
        fn write(&mut self, _buf: &[u8]) -> io::Result<usize> {
            Err(io::ErrorKind::BrokenPipe.into())
        }

        fn flush(&mut self) -> io::Result<()> {
            Err(io::ErrorKind::BrokenPipe.into())
        }
    }

    fn event(seq: u64) -> Event {
        Event {
            kind: EventKind::LineCross {
                dir: Direction::AToB,
            },
            rule: 1,
            rule_name: "door".to_string(),
            track: 7,
            class: 0,
            seq,
            at: Instant::now(),
        }
    }

    #[test]
    fn emit_sink_writes_lines_in_order() {
        let bytes = Arc::new(Mutex::new(Vec::new()));
        let sink = EventSink::spawn(Shared(bytes.clone()), WallClock::now()).unwrap();
        let sender = sink.sender();
        for seq in 1..=3 {
            sender.send(event(seq));
        }
        let dropped = sink.dropped();
        drop(sender);
        drop(sink);

        let text = String::from_utf8(bytes.lock().unwrap().clone()).unwrap();
        let seqs: Vec<u64> = text
            .lines()
            .map(|line| {
                let value: serde_json::Value = serde_json::from_str(line).unwrap();
                value["seq"].as_u64().unwrap()
            })
            .collect();
        assert_eq!(seqs, [1, 2, 3]);
        assert!(text.ends_with('\n'));
        assert_eq!(dropped.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn emit_sender_counts_drops_when_full() {
        // Nobody reads `_rx`, so the queue fills after two events.
        let (tx, _rx) = sync_channel(2);
        let dropped = Arc::new(AtomicU64::new(0));
        let sender = sender_for(tx, dropped.clone());
        for seq in 1..=4 {
            sender.send(event(seq));
        }
        assert_eq!(dropped.load(Ordering::Relaxed), 2);
    }

    #[test]
    fn emit_sink_survives_write_errors() {
        let sink = EventSink::spawn(BrokenPipe, WallClock::now()).unwrap();
        let sender = sink.sender();
        for seq in 1..=3 {
            sender.send(event(seq));
        }
        let dropped = sink.dropped();
        drop(sender);
        drop(sink); // joins: must not hang

        assert_eq!(dropped.load(Ordering::Relaxed), 3);
    }
}
