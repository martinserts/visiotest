//! The rules the user draws: lines and zones, and the `rules.json` file they live in.
//!
//! A rule is plain data: its geometry, the box [`Anchor`] that does the crossing, the
//! classes it counts and, for a zone, how long a stay counts as dwelling. The UI keeps
//! one copy (the editor changes it, and it is saved to `rules.json`), the detector
//! thread another (the engine runs on it).
//!
//! **Normalised coordinates.** Points are stored as fractions of the frame, `0..=1` on
//! each axis, not in pixels: the camera's resolution is only known at runtime, and the
//! UI draws the video into a rect of any size. The engine converts to frame pixels and
//! the editor to screen points (see [`super::geometry::to_pixels`]).
//!
//! **Ids.** Each rule gets an id, 1, 2, 3, … in the order it was added, never reused,
//! even after a rule is deleted or the file is loaded again. Events carry the id, so a
//! deleted "line 2" and a new "line 3" can never be mixed up.
//!
//! **The file** looks like this (classes are COCO names, since JSON readers lose the
//! precision of a 128-bit bitset; `anchor` may be left out and defaults to
//! `bottom_center`):
//!
//! ```json
//! {
//!   "version": 1,
//!   "lines": [
//!     { "id": 1, "name": "door", "a": { "x": 0.5, "y": 0.1 }, "b": { "x": 0.5, "y": 0.9 },
//!       "anchor": "bottom_center", "classes": ["person"] }
//!   ],
//!   "zones": [
//!     { "id": 2, "name": "desk", "points": [{ "x": 0.1, "y": 0.6 }, …],
//!       "anchor": "center", "classes": ["person"], "dwell_s": 10.0 }
//!   ]
//! }
//! ```
//!
//! Loading checks everything (the version, coordinates in `0..=1`, enough zone points,
//! the dwell range, known class names, unique ids) and names the rule in the error, so a
//! hand-edited file says exactly what is wrong with it. Saving writes a temporary file
//! and renames it over the old one, so a crash never leaves half a rules file.

mod file;

use std::io::ErrorKind;
use std::path::Path;
use std::time::Duration;

use anyhow::{Context, anyhow};
use serde::{Deserialize, Serialize};

use super::geometry::Point;
use crate::detect::classes::{ClassFilter, Group};

/// A rule's identity: 1, 2, 3, … never reused within one [`Rules`].
pub type RuleId = u32;

/// The point of a box that does the crossing (see [`super::geometry::anchor_point`]).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Anchor {
    /// The middle of the box's bottom edge, the "feet". The right choice when the floor
    /// is visible: a line on the image of the floor is then a line on the floor.
    #[default]
    BottomCenter,
    /// The middle of the box, for a desk view where the frame cuts people off.
    Center,
}

/// A tripwire: counts objects whose anchor crosses the segment `a`–`b`.
#[derive(Clone, Debug, PartialEq)]
pub struct LineRule {
    pub id: RuleId,
    /// "line 1" by default; editable.
    pub name: String,
    /// Normalised. The direction A→B is defined by `a → b`: side A is on your right when
    /// you stand on `a` and face `b` (see [`super::geometry::signed_distance`]).
    pub a: Point,
    pub b: Point,
    pub anchor: Anchor,
    /// Which classes this rule counts. People by default.
    pub classes: ClassFilter,
}

impl LineRule {
    /// Swaps `a` and `b`, which swaps the sides: what counted A→B now counts B→A.
    pub fn flip(&mut self) {
        std::mem::swap(&mut self.a, &mut self.b);
    }
}

/// A polygon: reports objects entering, dwelling in and leaving it.
#[derive(Clone, Debug, PartialEq)]
pub struct ZoneRule {
    pub id: RuleId,
    /// "zone 1" by default; editable.
    pub name: String,
    /// Normalised corners, [`MIN_ZONE_POINTS`]`..=`[`MAX_ZONE_POINTS`] of them.
    pub points: Vec<Point>,
    pub anchor: Anchor,
    /// Which classes this rule counts. People by default.
    pub classes: ClassFilter,
    /// A stay this long fires one `zone_dwell` event. Within [`MIN_DWELL`]`..=`[`MAX_DWELL`].
    pub dwell_after: Duration,
}

/// A polygon needs at least three corners to enclose anything.
pub const MIN_ZONE_POINTS: usize = 3;
/// Enough for any shape drawn by hand, and keeps the per-frame point-in-polygon and
/// convexity checks small.
pub const MAX_ZONE_POINTS: usize = 32;
/// Long enough that walking through a zone isn't dwelling, short enough to see it work.
pub const DEFAULT_DWELL: Duration = Duration::from_secs(10);
/// Below a second, "dwell" would fire for anyone passing through (the zone's enter
/// already waits 300 ms).
pub const MIN_DWELL: Duration = Duration::from_secs(1);
/// Two minutes: the UI slider's top end.
pub const MAX_DWELL: Duration = Duration::from_secs(120);

/// Every rule, lines first. Plain data: the UI owns one copy, the detector another.
#[derive(Clone, Debug, PartialEq)]
pub struct Rules {
    pub lines: Vec<LineRule>,
    pub zones: Vec<ZoneRule>,
    /// The id the next added rule gets.
    next_id: RuleId,
}

impl Default for Rules {
    fn default() -> Self {
        Self {
            lines: Vec::new(),
            zones: Vec::new(),
            next_id: 1,
        }
    }
}

impl Rules {
    /// Adds a line from `a` to `b` (normalised) with the default settings; returns its id.
    pub fn add_line(&mut self, a: Point, b: Point) -> RuleId {
        let id = self.take_id();
        self.lines.push(LineRule {
            id,
            name: format!("line {id}"),
            a,
            b,
            anchor: Anchor::default(),
            classes: ClassFilter::only(Group::People),
        });
        id
    }

    /// Adds a zone (normalised corners; the caller guarantees at least 3) with the
    /// default settings; returns its id.
    pub fn add_zone(&mut self, points: Vec<Point>) -> RuleId {
        let id = self.take_id();
        self.zones.push(ZoneRule {
            id,
            name: format!("zone {id}"),
            points,
            anchor: Anchor::default(),
            classes: ClassFilter::only(Group::People),
            dwell_after: DEFAULT_DWELL,
        });
        id
    }

    /// Removes rule `id`; false if there was none. Its id is not reused.
    pub fn remove(&mut self, id: RuleId) -> bool {
        if let Some(i) = self.lines.iter().position(|line| line.id == id) {
            self.lines.remove(i);
            return true;
        }
        if let Some(i) = self.zones.iter().position(|zone| zone.id == id) {
            self.zones.remove(i);
            return true;
        }
        false
    }

    pub fn line(&self, id: RuleId) -> Option<&LineRule> {
        self.lines.iter().find(|line| line.id == id)
    }

    pub fn zone(&self, id: RuleId) -> Option<&ZoneRule> {
        self.zones.iter().find(|zone| zone.id == id)
    }

    pub fn line_mut(&mut self, id: RuleId) -> Option<&mut LineRule> {
        self.lines.iter_mut().find(|line| line.id == id)
    }

    pub fn zone_mut(&mut self, id: RuleId) -> Option<&mut ZoneRule> {
        self.zones.iter_mut().find(|zone| zone.id == id)
    }

    /// The name of rule `id`, line or zone.
    pub fn name(&self, id: RuleId) -> Option<&str> {
        if let Some(line) = self.line(id) {
            return Some(&line.name);
        }
        self.zone(id).map(|zone| zone.name.as_str())
    }

    /// The number of rules, lines and zones together.
    pub fn len(&self) -> usize {
        self.lines.len() + self.zones.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// "2 lines, 1 zone" for the startup line; "no rules" when empty.
    pub fn summary(&self) -> String {
        if self.is_empty() {
            return "no rules".to_string();
        }
        let mut parts = Vec::new();
        if !self.lines.is_empty() {
            parts.push(count_of(self.lines.len(), "line"));
        }
        if !self.zones.is_empty() {
            parts.push(count_of(self.zones.len(), "zone"));
        }
        parts.join(", ")
    }

    /// The rules as pretty-printed `rules.json` text, with `"version": 1`.
    pub fn to_json(&self) -> String {
        file::to_json(self)
    }

    /// Parses and checks `rules.json` text. The error names the rule at fault, e.g.
    /// `zone "desk" (id 2): needs at least 3 points, has 2`.
    pub fn from_json(text: &str) -> Result<Rules, String> {
        file::from_json(text)
    }

    /// Reads the rules file at `path`; `Ok(None)` when the file doesn't exist.
    pub fn load(path: &Path) -> anyhow::Result<Option<Rules>> {
        let text = match std::fs::read_to_string(path) {
            Ok(text) => text,
            Err(e) if e.kind() == ErrorKind::NotFound => return Ok(None),
            Err(e) => return Err(e).with_context(|| format!("reading {}", path.display())),
        };
        let rules = Rules::from_json(&text).map_err(|e| anyhow!("{}: {e}", path.display()))?;
        Ok(Some(rules))
    }

    /// Writes `path.tmp`, then renames it over `path`, so a crash never leaves half a file.
    pub fn save(&self, path: &Path) -> anyhow::Result<()> {
        // A rename within one directory is atomic: anyone reading `path` sees either the
        // old file or the new one, never a half-written one.
        let tmp = path.with_extension("json.tmp");
        std::fs::write(&tmp, self.to_json())
            .with_context(|| format!("writing {}", tmp.display()))?;
        std::fs::rename(&tmp, path)
            .with_context(|| format!("renaming {} to {}", tmp.display(), path.display()))
    }

    /// Hands out the next id.
    fn take_id(&mut self) -> RuleId {
        let id = self.next_id;
        self.next_id += 1;
        id
    }
}

/// "1 line", "2 lines".
fn count_of(n: usize, noun: &str) -> String {
    if n == 1 {
        format!("1 {noun}")
    } else {
        format!("{n} {noun}s")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn pt(x: f32, y: f32) -> Point {
        Point::new(x, y)
    }

    fn triangle() -> Vec<Point> {
        vec![pt(0.1, 0.1), pt(0.5, 0.1), pt(0.3, 0.5)]
    }

    /// A unique path under the system temp dir, so parallel tests never collide.
    fn temp_path(test: &str) -> PathBuf {
        std::env::temp_dir().join(format!(
            "visiotest-rules-{}-{test}.json",
            std::process::id()
        ))
    }

    fn remove_with_tmp(path: &Path) {
        let _ = std::fs::remove_file(path);
        let _ = std::fs::remove_file(path.with_extension("json.tmp"));
    }

    #[test]
    fn events_rules_ids_never_reused() {
        let mut rules = Rules::default();
        assert!(rules.is_empty());
        assert_eq!(rules.summary(), "no rules");

        assert_eq!(rules.add_line(pt(0.5, 0.0), pt(0.5, 1.0)), 1);
        assert_eq!(rules.add_zone(triangle()), 2);
        assert_eq!(rules.name(1), Some("line 1"));
        assert_eq!(rules.name(2), Some("zone 2"));
        assert_eq!(rules.summary(), "1 line, 1 zone");

        assert!(rules.remove(2));
        assert!(!rules.remove(2));
        assert_eq!(rules.add_zone(triangle()), 3);
        assert_eq!(rules.name(3), Some("zone 3"));
        assert_eq!(rules.name(2), None);
        assert_eq!(rules.len(), 2);

        rules.add_line(pt(0.0, 0.5), pt(1.0, 0.5));
        assert_eq!(rules.summary(), "2 lines, 1 zone");

        let line = &rules.lines[0];
        assert_eq!(line.anchor, Anchor::BottomCenter);
        assert_eq!(line.classes, ClassFilter::only(Group::People));
        assert_eq!(rules.zones[0].dwell_after, DEFAULT_DWELL);
    }

    #[test]
    fn events_rules_json_round_trip() {
        let mut rules = Rules::default();
        let line = rules.add_line(pt(0.5, 0.1), pt(0.5, 0.9));
        let zone = rules.add_zone(triangle());

        let mut person_and_car = ClassFilter::none();
        person_and_car.set(0, true); // person
        person_and_car.set(2, true); // car
        let l = rules.line_mut(line).unwrap();
        l.name = "door".to_string();
        l.anchor = Anchor::Center;
        l.classes = person_and_car;
        let z = rules.zone_mut(zone).unwrap();
        z.name = "desk".to_string();
        z.classes = person_and_car;
        z.dwell_after = Duration::from_secs(30);

        let text = rules.to_json();
        assert!(text.contains("\"version\": 1"), "{text}");
        assert!(text.contains("\"person\""), "{text}");
        let mut back = Rules::from_json(&text).unwrap();
        assert_eq!(back, rules);
        // The next id continues after the highest one in the file.
        assert_eq!(back.add_line(pt(0.0, 0.0), pt(1.0, 1.0)), 3);
    }

    #[test]
    fn events_rules_flip_swaps_the_sides() {
        let mut rules = Rules::default();
        let id = rules.add_line(pt(0.5, 0.1), pt(0.5, 0.9));
        let line = rules.line_mut(id).unwrap();
        line.flip();
        assert_eq!((line.a, line.b), (pt(0.5, 0.9), pt(0.5, 0.1)));
        // A point on screen-left was side A; after the flip it is side B.
        let left = pt(0.4, 0.5);
        assert!(crate::events::geometry::signed_distance(line.a, line.b, left) < 0.0);
    }

    #[test]
    fn events_rules_load_missing_is_none() {
        let path = temp_path("events_rules_load_missing_is_none");
        remove_with_tmp(&path);
        assert!(Rules::load(&path).unwrap().is_none());
    }

    #[test]
    fn events_rules_save_then_load() {
        let path = temp_path("events_rules_save_then_load");
        let mut rules = Rules::default();
        rules.add_line(pt(0.5, 0.0), pt(0.5, 1.0));
        rules.add_zone(triangle());

        rules.save(&path).unwrap();
        let loaded = Rules::load(&path).unwrap();
        let tmp_left = path.with_extension("json.tmp").exists();
        remove_with_tmp(&path);

        assert_eq!(loaded, Some(rules));
        assert!(!tmp_left, "the temporary file is renamed away");
    }
}
