//! The fixed catalogue of synthetic scenes, one difficulty each.
//!
//! | scene | what it tests | expected with every-frame detection |
//! |---|---|---|
//! | `walk` | one person, 100×300 px, crossing the frame at 480 px/s (≈ 4 s) | 0 ID switches, 1 ID |
//! | `pole` | the same, behind a 200 px pole: hidden ≈ 0.31 s, half-hidden on both sides | 0 ID switches |
//! | `wall` | the same, behind a 900 px wall: hidden ≈ 1.8 s, longer than a lost track lives | a new ID afterwards: 2 IDs |
//! | `crossing` | two people, 100×300 and 140×420, passing in opposite directions | 0 ID switches |
//! | `still` | three cups standing still for 10 s, tiny jitter | 0 ID switches |
//! | `crowd` | six random walkers in depth lanes, two poles, 0.5 false positives/s | reported only |
//! | `stop` | one person who stops dead for 1.5 s, then walks back | 0 ID switches; paced modes reported |
//!
//! The hidden times follow from the visibility rule in [`super`]: a 100 px walker is
//! invisible while more than 75 px of it are covered. Behind a 200 px pole that lasts
//! for 200 + 100 − 2 · 75 = 150 px of travel, 150 / 480 ≈ 0.31 s; behind a 900 px wall
//! for 850 px, ≈ 1.8 s.
//!
//! Every scene except `stop` moves at constant speed, the Kalman filter's own model, so
//! they flatter a filter that is slow to change its mind. `stop` is the counterweight:
//! it shows what that slowness costs.

use super::{DEFAULT_BOX_NOISE, FRAME_H, FRAME_W, Occluder, Rng, Scene, Walker};

/// COCO class ids (`detect::classes::CLASS_NAMES`).
const PERSON: u8 = 0;
const CUP: u8 = 41;

/// Speed of the single-walker scenes: a brisk walk across a 1080p frame in 4 s.
const WALK_SPEED: f32 = 480.0;
/// The single walker's size: a person ~3 m from the camera.
const WALKER_SIZE: (f32, f32) = (100.0, 300.0);

/// Every scene, in the order the report prints them.
pub fn scenes() -> Vec<Scene> {
    vec![walk(), pole(), wall(), crossing(), still(), crowd(), stop()]
}

/// A person walking at constant velocity `vx`, centred at `start_cx` at time 0, there
/// for the whole scene.
fn person(object: u32, start_cx: f32, cy: f32, vx: f32, size: (f32, f32)) -> Walker {
    Walker {
        object,
        start: (start_cx, cy),
        velocity: (vx, 0.0),
        size,
        enter_s: 0.0,
        leave_s: f32::INFINITY,
        class: PERSON,
    }
}

/// The single walker of `walk`, `pole` and `wall`: starts just off the left edge.
fn lone_walker() -> Walker {
    person(
        0,
        -WALKER_SIZE.0 / 2.0,
        FRAME_H / 2.0,
        WALK_SPEED,
        WALKER_SIZE,
    )
}

/// A scene with default noise and no false positives.
fn scene(name: &'static str, duration_s: f32, walkers: Vec<Walker>, seed: u64) -> Scene {
    Scene {
        name,
        duration_s,
        walkers,
        occluders: Vec::new(),
        seed,
        box_noise: DEFAULT_BOX_NOISE,
        false_positives_per_s: 0.0,
    }
}

/// An occluder `width` px wide, centred in the frame.
fn centred(width: f32) -> Occluder {
    Occluder {
        x0: (FRAME_W - width) / 2.0,
        x1: (FRAME_W + width) / 2.0,
    }
}

fn walk() -> Scene {
    // (1920 + 100) px / 480 px/s ≈ 4.2 s until the walker is gone.
    scene("walk", 4.5, vec![lone_walker()], 1)
}

fn pole() -> Scene {
    Scene {
        occluders: vec![centred(200.0)],
        ..scene("pole", 4.5, vec![lone_walker()], 2)
    }
}

fn wall() -> Scene {
    Scene {
        occluders: vec![centred(900.0)],
        ..scene("wall", 4.5, vec![lone_walker()], 3)
    }
}

fn crossing() -> Scene {
    // A walks right, B (nearer, so bigger and lower in the frame) walks left. They
    // pass each other at about 2.4 s; their boxes overlap by IoU up to ≈ 0.3 then.
    let a = person(0, -50.0, 450.0, WALK_SPEED, (100.0, 300.0));
    let b = person(1, FRAME_W + 70.0, 600.0, -360.0, (140.0, 420.0));
    scene("crossing", 6.0, vec![a, b], 4)
}

fn still() -> Scene {
    let cup = |object: u32, cx: f32, cy: f32| Walker {
        object,
        start: (cx, cy),
        velocity: (0.0, 0.0),
        size: (60.0, 80.0),
        enter_s: 0.0,
        leave_s: f32::INFINITY,
        class: CUP,
    };
    Scene {
        // 1 % = 0.6 px: a steady camera on objects that don't move.
        box_noise: 0.01,
        ..scene(
            "still",
            10.0,
            vec![
                cup(0, 500.0, 700.0),
                cup(1, 900.0, 720.0),
                cup(2, 1300.0, 690.0),
            ],
            5,
        )
    }
}

fn crowd() -> Scene {
    const WALKERS: u32 = 6;
    // A separate seed for the layout, so changing the noise seed keeps the walkers.
    let mut rng = Rng::new(60);
    let walkers = (0..WALKERS)
        .map(|lane| random_walker(lane, &mut rng))
        .collect();
    Scene {
        occluders: vec![
            Occluder {
                x0: 600.0,
                x1: 700.0,
            },
            Occluder {
                x0: 1250.0,
                x1: 1400.0,
            },
        ],
        false_positives_per_s: 0.5,
        ..scene("crowd", 15.0, walkers, 6)
    }
}

/// A person walking along `lane` (0 = farthest from the camera), at a random speed,
/// entering at a random time in the first 6 s.
///
/// The lanes stand in for depth, because the scenes don't model people hiding each
/// other: two random walkers at the same depth and speed would overlap almost exactly
/// for seconds, and then no tracker (and no ground truth matching) could tell them apart.
/// So each lane is nearer (bigger, lower in the frame) than the one before, and
/// neighbouring lanes walk in opposite directions, like a busy pavement.
fn random_walker(lane: u32, rng: &mut Rng) -> Walker {
    // Farthest lane ~200 px tall, nearest ~500 px, ± a little per person.
    let h = 200.0 + 60.0 * lane as f32 + rng.range(-20.0, 20.0);
    // People's boxes are about a third as wide as they are tall.
    let w = h * 0.35;
    // Nearer people stand lower in the frame: feet from y ≈ 520 (far) to ≈ 970 (near).
    let feet = 520.0 + 1.5 * (h - 200.0);
    let speed = rng.range(200.0, 450.0);
    let enter_s = rng.range(0.0, 6.0);
    // Where it must be at time 0 to stand just off its entry edge at `enter_s`.
    let (edge, vx) = if lane.is_multiple_of(2) {
        (-w / 2.0, speed)
    } else {
        (FRAME_W + w / 2.0, -speed)
    };
    Walker {
        enter_s,
        ..person(lane, edge - vx * enter_s, feet - h / 2.0, vx, (w, h))
    }
}

fn stop() -> Scene {
    // One object, three straight pieces: walk right, stand at x 910 from 2 s to 3.5 s,
    // walk back left. Each piece is a `Walker` with the same `object`, so to the
    // ground truth it is one person changing speed in an instant.
    let walking = lone_walker();
    let stop_x = walking.start.0 + WALK_SPEED * 2.0;
    let pieces = vec![
        Walker {
            leave_s: 2.0,
            ..walking
        },
        Walker {
            start: (stop_x, walking.start.1),
            velocity: (0.0, 0.0),
            enter_s: 2.0,
            leave_s: 3.5,
            ..walking
        },
        Walker {
            // Back at `stop_x` at 3.5 s, walking left.
            start: (stop_x + WALK_SPEED * 3.5, walking.start.1),
            velocity: (-WALK_SPEED, 0.0),
            enter_s: 3.5,
            ..walking
        },
    ];
    scene("stop", 6.0, pieces, 7)
}
