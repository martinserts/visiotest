//! Tests for the scenes and the runner, and the benchmark report (an ignored test).

use super::*;

/// The confidence slider's default: detections ≥ this are ByteTrack's "high" ones.
const HIGH_CONF: f32 = 0.35;

/// The pacing modes the report compares.
const MODES: [DetectInterval; 4] = [
    DetectInterval::EveryFrame,
    DetectInterval::Fixed(Duration::from_millis(100)),
    DetectInterval::Fixed(Duration::from_millis(250)),
    DetectInterval::Auto,
];

fn scene_named(name: &str) -> Scene {
    scenes()
        .into_iter()
        .find(|s| s.name == name)
        .unwrap_or_else(|| panic!("no scene {name}"))
}

/// A 100 × 300 box with its left edge at `x0`.
fn walker_box(x0: f32) -> BBox {
    BBox::new(x0, 400.0, x0 + 100.0, 700.0)
}

#[test]
fn sim_visibility_rules() {
    let pole = [Occluder {
        x0: 1000.0,
        x1: 1200.0,
    }];
    let mut rng = Rng::new(1);
    let see = |x0: f32, rng: &mut Rng| {
        let seen = sighting(walker_box(x0), &pole);
        (seen, seen.and_then(|s| detect(s, 0, 0.0, rng)))
    };

    // Clear of the pole: all of it, confident.
    let (seen, det) = see(500.0, &mut rng);
    assert_eq!(seen.unwrap().visible, 1.0);
    let det = det.expect("a clear walker is detected");
    assert_eq!(det.bbox, walker_box(500.0), "no noise: the exact box");
    assert!((0.75..=0.85).contains(&det.score), "{}", det.score);

    // Half behind the pole (x 950..1050, pole from 1000): the left half, weak.
    let (seen, det) = see(950.0, &mut rng);
    assert_eq!(seen.unwrap().visible, 0.5);
    let det = det.expect("a half-hidden walker gets a weak box");
    assert_eq!(det.bbox, BBox::new(950.0, 400.0, 1000.0, 700.0));
    assert!((0.15..=0.25).contains(&det.score), "{}", det.score);

    // Fully behind it: seen by nobody.
    let (seen, det) = see(1050.0, &mut rng);
    assert_eq!((seen, det), (None, None));

    // A sliver (20 % shows): a sighting, but no detection.
    let (seen, det) = see(920.0, &mut rng);
    assert!((seen.unwrap().visible - 0.8).abs() < 1e-6);
    assert!(det.is_some());
    let (seen, det) = see(980.0, &mut rng);
    assert!((seen.unwrap().visible - 0.2).abs() < 1e-6);
    assert_eq!(det, None);
}

#[test]
fn sim_frame_edge_and_thin_pole() {
    // Half off the left edge of the frame: the in-frame half.
    let seen = sighting(walker_box(-50.0), &[]).unwrap();
    assert_eq!((seen.bbox.x0, seen.bbox.x1, seen.visible), (0.0, 50.0, 0.5));
    // A 20 px pole across the middle: the box still spans the whole walker.
    let thin = [Occluder {
        x0: 540.0,
        x1: 560.0,
    }];
    let seen = sighting(walker_box(500.0), &thin).unwrap();
    assert_eq!((seen.bbox.x0, seen.bbox.x1), (500.0, 600.0));
    assert!((seen.visible - 0.8).abs() < 1e-6);
}

#[test]
fn sim_is_deterministic() {
    let crowd = scene_named("crowd");
    for mode in MODES {
        assert_eq!(run(&crowd, mode, HIGH_CONF), run(&crowd, mode, HIGH_CONF));
    }
}

#[test]
fn sim_walk_pole_crossing_still_have_no_id_switches() {
    for name in ["walk", "pole", "crossing", "still"] {
        let r = run(&scene_named(name), DetectInterval::EveryFrame, HIGH_CONF);
        assert_eq!(r.score.id_switches, 0, "{r:?}");
        assert_eq!(r.ids_per_object, 1.0, "{r:?}");
    }
}

#[test]
fn sim_stop_keeps_its_id_every_frame() {
    // A stiff filter coasts well through occlusions but is slow to notice a stop; with
    // a detection every frame it must still keep the id (see `KalmanConfig::default`).
    let r = run(&scene_named("stop"), DetectInterval::EveryFrame, HIGH_CONF);
    assert_eq!(r.score.id_switches, 0, "{r:?}");
}

#[test]
fn sim_pole_survives_with_auto() {
    let r = run(&scene_named("pole"), DetectInterval::Auto, HIGH_CONF);
    assert_eq!(r.score.id_switches, 0, "{r:?}");
    assert_eq!(r.ids_per_object, 1.0, "{r:?}");
}

#[test]
fn sim_wall_documents_the_limit() {
    // Hidden ≈ 1.8 s, longer than a lost track is kept (1 s): it comes back as a new id.
    let r = run(&scene_named("wall"), DetectInterval::EveryFrame, HIGH_CONF);
    assert_eq!(r.ids_per_object, 2.0, "{r:?}");
    assert_eq!(r.score.id_switches, 1, "{r:?}");
}

#[test]
fn sim_auto_saves_compute_on_still() {
    let still = scene_named("still");
    let every = run(&still, DetectInterval::EveryFrame, HIGH_CONF);
    let auto = run(&still, DetectInterval::Auto, HIGH_CONF);
    assert_eq!(every.detector_runs, every.frames);
    assert!(
        auto.detector_runs * 4 <= every.detector_runs,
        "auto ran {} of {} frames",
        auto.detector_runs,
        every.detector_runs
    );
    assert_eq!(auto.score.id_switches, 0, "{auto:?}");
}

/// One row of the report, fixed-width so the columns line up.
fn report_row(r: &RunReport) -> String {
    let runs_pct = 100.0 * r.detector_runs as f32 / r.frames as f32;
    let iou = r
        .score
        .mean_iou()
        .map_or_else(|| "—".to_string(), |m| format!("{m:.3}"));
    format!(
        "{:<9} {:<6} {:>5} {:>6.1} {:>5} {:>7.2} {:>6.3} {:>6}",
        r.scene,
        r.mode.to_string(),
        r.detector_runs,
        runs_pct,
        r.score.id_switches,
        r.ids_per_object,
        r.score.mota(),
        iou
    )
}

/// The benchmark: every scene × every pacing mode. `runs%` is the share of frames the
/// detector ran on (the compute spent), and every frame is scored, so the predicted
/// boxes between runs count. `IDs/obj` 0 means no track was ever confirmed.
#[test]
#[ignore = "benchmark: cargo test --release sim_report -- --ignored --nocapture"]
fn sim_report() {
    println!(
        "{:<9} {:<6} {:>5} {:>6} {:>5} {:>7} {:>6} {:>6}",
        "scene", "mode", "runs", "runs%", "IDSW", "IDs/obj", "MOTA", "IoU"
    );
    for scene in scenes() {
        for mode in MODES {
            println!("{}", report_row(&run(&scene, mode, HIGH_CONF)));
        }
    }
}
