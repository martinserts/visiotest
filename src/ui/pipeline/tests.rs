use std::time::Duration;

use super::*;
use crate::app::SourceInfo;
use crate::stats::Summary;

/// A detector result exists: every health rule past Idle needs one.
fn result() -> Option<Summary> {
    Some(Summary {
        last: 16.0,
        p50: 16.0,
        p95: 18.0,
    })
}

#[test]
fn stages_are_in_pipeline_order_with_keys_one_to_five() {
    let titles = Stage::ALL.map(Stage::title);
    assert_eq!(titles, ["Camera", "Detect", "Track", "Events", "Latency"]);
    assert_eq!(Stage::ALL.map(Stage::key), ['1', '2', '3', '4', '5']);
}

#[test]
fn only_latency_is_not_a_stage() {
    let stages = Stage::ALL.map(Stage::is_stage);
    assert_eq!(stages, [true, true, true, true, false]);
}

#[test]
fn every_blurb_is_one_sentence() {
    for stage in Stage::ALL {
        let blurb = stage.blurb();
        assert!(!blurb.is_empty(), "{stage:?}");
        assert!(blurb.ends_with('.'), "{stage:?}: {blurb}");
    }
}

/// A 1080p30 camera, for the strip's texts.
fn camera() -> SourceInfo {
    SourceInfo {
        name: "cam".into(),
        format: "YUYV".into(),
        width: 1920,
        height: 1080,
        fps: 30,
    }
}

/// A status with no reason, for comparing whole `Status`es.
fn plain(health: Health) -> Status {
    Status {
        health,
        reason: None,
    }
}

/// A Warn's reason, which must exist: a Warn without one is a bug.
fn reason(status: &Status) -> &str {
    assert_eq!(status.health, Health::Warn);
    status.reason.as_deref().expect("a Warn has a reason")
}

#[test]
fn camera_status_rules() {
    let idle = TelemetrySnapshot::default();
    assert_eq!(camera_status(&idle, 30), plain(Health::Idle));
    let running = TelemetrySnapshot {
        seq: 10,
        camera_fps: 29.9,
        ..Default::default()
    };
    assert_eq!(camera_status(&running, 30), plain(Health::Ok));
    let slow = TelemetrySnapshot {
        camera_fps: 24.1,
        ..running
    };
    assert_eq!(
        reason(&camera_status(&slow, 30)),
        "Camera at 24.1 fps, below 90 % of 30"
    );
    let bad = TelemetrySnapshot {
        bad_buffers: 2,
        ..running
    };
    assert_eq!(reason(&camera_status(&bad, 30)), "2 bad buffers");
}

#[test]
fn camera_status_two_problems_are_two_lines() {
    let both = TelemetrySnapshot {
        seq: 10,
        camera_fps: 20.0,
        bad_buffers: 1,
        ..Default::default()
    };
    let status = camera_status(&both, 30);
    let lines: Vec<&str> = reason(&status).lines().collect();
    assert_eq!(
        lines,
        ["Camera at 20.0 fps, below 90 % of 30", "1 bad buffers"]
    );
}

#[test]
fn detect_status_rules() {
    let none = TelemetrySnapshot::default();
    assert_eq!(detect_status(&none, false), plain(Health::Off));
    assert_eq!(detect_status(&none, true), plain(Health::Idle));

    let every_frame = TelemetrySnapshot {
        detect_total: result(),
        camera_fps: 30.0,
        detector_fps: 29.9,
        detect_interval: Some(Duration::ZERO),
        ..Default::default()
    };
    assert_eq!(detect_status(&every_frame, true), plain(Health::Ok));
    let errors = TelemetrySnapshot {
        det_errors: 2,
        ..every_frame
    };
    assert_eq!(reason(&detect_status(&errors, true)), "2 detector errors");

    // 20 fps for a 30 fps camera: behind when every frame is asked for…
    let behind = TelemetrySnapshot {
        detector_fps: 20.0,
        ..every_frame
    };
    assert_eq!(
        reason(&detect_status(&behind, true)),
        "Detector at 20.0 fps, behind the camera's 30.0"
    );
    // …but slower on purpose when paced.
    let paced = TelemetrySnapshot {
        detect_interval: Some(Duration::from_millis(250)),
        ..behind
    };
    assert_eq!(detect_status(&paced, true), plain(Health::Ok));

    let both = TelemetrySnapshot {
        det_errors: 1,
        ..behind
    };
    assert_eq!(
        reason(&detect_status(&both, true)),
        "1 detector errors\nDetector at 20.0 fps, behind the camera's 30.0"
    );
}

#[test]
fn track_status_rules() {
    let none = TelemetrySnapshot::default();
    assert_eq!(track_status(&none, false), plain(Health::Off));
    assert_eq!(track_status(&none, true), plain(Health::Idle));
    let running = TelemetrySnapshot {
        detect_total: result(),
        tracks_lost: 3,
        ..Default::default()
    };
    assert_eq!(track_status(&running, true), plain(Health::Ok));
}

#[test]
fn events_status_rules() {
    let none = TelemetrySnapshot::default();
    assert_eq!(events_status(&none, false, None), plain(Health::Off));
    // No Idle: rules exist before any result.
    assert_eq!(events_status(&none, true, None), plain(Health::Ok));
    assert_eq!(
        reason(&events_status(&none, true, Some("disk full"))),
        "Rules not saved: disk full"
    );
    let dropped = TelemetrySnapshot {
        events_dropped: 3,
        ..Default::default()
    };
    assert_eq!(
        reason(&events_status(&dropped, true, None)),
        "3 events dropped"
    );
    assert_eq!(
        reason(&events_status(&dropped, true, Some("disk full"))),
        "3 events dropped\nRules not saved: disk full"
    );
}

#[test]
fn strip_state_defaults_to_idle_without_reasons() {
    let state = StripState::default();
    for status in [&state.camera, &state.detect, &state.track, &state.events] {
        assert_eq!(*status, plain(Health::Idle));
    }
}

#[test]
fn rows_take_values_from_the_text_and_latency_stays_neutral() {
    let text = DashText::new(&camera(), false);
    let failed = SaveStatus::Failed("disk full".into());
    let state = StripState::new(&TelemetrySnapshot::default(), 30, false, &failed);
    let rows = rows(&text, &state);
    assert_eq!(rows.map(|r| r.stage), Stage::ALL);
    assert_eq!(rows[1].value, "off");
    let health = rows.map(|r| r.health);
    let off = Health::Off;
    assert_eq!(health, [Health::Idle, off, off, off, Health::Idle]);
    // Off stages have nothing to explain, even with a failed save.
    assert!(rows.iter().all(|r| r.reason.is_none()));
}

#[test]
fn rows_carry_the_warn_reason() {
    let text = DashText::new(&camera(), true);
    let failed = SaveStatus::Failed("disk full".into());
    let state = StripState::new(&TelemetrySnapshot::default(), 30, true, &failed);
    let rows = rows(&text, &state);
    assert_eq!(rows[3].health, Health::Warn);
    assert_eq!(rows[3].reason, Some("Rules not saved: disk full"));
    assert_eq!(rows[0].reason, None);
}
