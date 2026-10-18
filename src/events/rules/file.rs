//! The `rules.json` format: the file's own types, and the checks a loaded file must pass.
//!
//! The file has its own small structs (a *DTO*, data transfer object) rather than
//! putting serde on [`Rules`] itself, so the in-memory types stay plain: a
//! [`ClassFilter`] bitset becomes a list of class names, and a `Duration` becomes
//! `dwell_s` in seconds.

use std::time::Duration;

use serde::{Deserialize, Serialize};

use super::{
    Anchor, LineRule, MAX_DWELL, MAX_ZONE_POINTS, MIN_DWELL, MIN_ZONE_POINTS, RuleId, Rules,
    ZoneRule,
};
use crate::detect::NUM_CLASSES;
use crate::detect::classes::{CLASS_NAMES, ClassFilter};
use crate::events::geometry::Point;

/// Bumped if the format ever changes in a way old files can't be read as.
const FILE_VERSION: u32 = 1;

#[derive(Serialize, Deserialize)]
struct RulesFile {
    version: u32,
    lines: Vec<LineFile>,
    zones: Vec<ZoneFile>,
}

#[derive(Serialize, Deserialize)]
struct LineFile {
    id: RuleId,
    name: String,
    a: Point,
    b: Point,
    #[serde(default)]
    anchor: Anchor,
    classes: Vec<String>,
}

#[derive(Serialize, Deserialize)]
struct ZoneFile {
    id: RuleId,
    name: String,
    points: Vec<Point>,
    #[serde(default)]
    anchor: Anchor,
    classes: Vec<String>,
    dwell_s: f32,
}

/// `rules` as pretty-printed JSON.
pub(super) fn to_json(rules: &Rules) -> String {
    let file = RulesFile {
        version: FILE_VERSION,
        lines: rules.lines.iter().map(line_to_file).collect(),
        zones: rules.zones.iter().map(zone_to_file).collect(),
    };
    // Plain structs, strings and numbers can't fail to serialise (serde_json writes a
    // non-finite float as `null`, which `from_json` then rejects).
    serde_json::to_string_pretty(&file).expect("rules always serialise")
}

/// Parses and checks `text`; the error names the rule at fault.
pub(super) fn from_json(text: &str) -> Result<Rules, String> {
    let file: RulesFile =
        serde_json::from_str(text).map_err(|e| format!("not a valid rules file: {e}"))?;
    if file.version != FILE_VERSION {
        return Err(format!(
            "rules file version {} isn't supported (expected {FILE_VERSION})",
            file.version
        ));
    }
    let mut rules = Rules::default();
    for line in file.lines {
        rules.lines.push(line_from_file(line)?);
    }
    for zone in file.zones {
        rules.zones.push(zone_from_file(zone)?);
    }
    check_unique_ids(&rules)?;
    // Continue after the highest id, so ids of deleted rules are never handed out again.
    let line_ids = rules.lines.iter().map(|line| line.id);
    let zone_ids = rules.zones.iter().map(|zone| zone.id);
    let max_id = line_ids.chain(zone_ids).max().unwrap_or(0);
    rules.next_id = max_id + 1;
    Ok(rules)
}

fn line_to_file(line: &LineRule) -> LineFile {
    LineFile {
        id: line.id,
        name: line.name.clone(),
        a: line.a,
        b: line.b,
        anchor: line.anchor,
        classes: class_names(line.classes),
    }
}

fn zone_to_file(zone: &ZoneRule) -> ZoneFile {
    ZoneFile {
        id: zone.id,
        name: zone.name.clone(),
        points: zone.points.clone(),
        anchor: zone.anchor,
        classes: class_names(zone.classes),
        dwell_s: zone.dwell_after.as_secs_f32(),
    }
}

fn line_from_file(line: LineFile) -> Result<LineRule, String> {
    let at_fault = |msg: String| format!("line \"{}\" (id {}): {msg}", line.name, line.id);
    check_id(line.id).map_err(at_fault)?;
    check_points(&[line.a, line.b]).map_err(at_fault)?;
    if line.a == line.b {
        return Err(at_fault("its two ends are the same point".to_string()));
    }
    let classes = class_filter(&line.classes).map_err(at_fault)?;
    Ok(LineRule {
        id: line.id,
        name: line.name,
        a: line.a,
        b: line.b,
        anchor: line.anchor,
        classes,
    })
}

fn zone_from_file(zone: ZoneFile) -> Result<ZoneRule, String> {
    let at_fault = |msg: String| format!("zone \"{}\" (id {}): {msg}", zone.name, zone.id);
    check_id(zone.id).map_err(at_fault)?;
    let n = zone.points.len();
    if n < MIN_ZONE_POINTS {
        let msg = format!("needs at least {MIN_ZONE_POINTS} points, has {n}");
        return Err(at_fault(msg));
    }
    if n > MAX_ZONE_POINTS {
        let msg = format!("may have at most {MAX_ZONE_POINTS} points, has {n}");
        return Err(at_fault(msg));
    }
    check_points(&zone.points).map_err(at_fault)?;
    let dwell_range = MIN_DWELL.as_secs_f32()..=MAX_DWELL.as_secs_f32();
    // `contains` is false for NaN too, so the `from_secs_f32` below can't panic.
    if !dwell_range.contains(&zone.dwell_s) {
        let msg = format!(
            "dwell_s {} is outside {}..={} s",
            zone.dwell_s,
            dwell_range.start(),
            dwell_range.end()
        );
        return Err(at_fault(msg));
    }
    let classes = class_filter(&zone.classes).map_err(at_fault)?;
    Ok(ZoneRule {
        id: zone.id,
        name: zone.name,
        points: zone.points,
        anchor: zone.anchor,
        classes,
        dwell_after: Duration::from_secs_f32(zone.dwell_s),
    })
}

/// Ids start at 1, like track ids.
fn check_id(id: RuleId) -> Result<(), String> {
    if id == 0 {
        return Err("ids start at 1".to_string());
    }
    Ok(())
}

/// Every coordinate is normalised: a finite number in `0..=1`.
fn check_points(points: &[Point]) -> Result<(), String> {
    let normalised = |v: f32| (0.0..=1.0).contains(&v); // false for NaN as well
    match points.iter().find(|p| !normalised(p.x) || !normalised(p.y)) {
        Some(p) => Err(format!(
            "point ({}, {}) is outside the frame (coordinates are 0..=1)",
            p.x, p.y
        )),
        None => Ok(()),
    }
}

/// Fails on the first id used by more than one rule, naming that rule.
fn check_unique_ids(rules: &Rules) -> Result<(), String> {
    let mut ids: Vec<RuleId> = rules.lines.iter().map(|l| l.id).collect();
    ids.extend(rules.zones.iter().map(|z| z.id));
    ids.sort_unstable();
    // After sorting, equal ids sit next to each other.
    match ids.windows(2).find(|pair| pair[0] == pair[1]) {
        Some(pair) => {
            let id = pair[0];
            let name = rules.name(id).unwrap_or_default();
            Err(format!(
                "rule \"{name}\" (id {id}): id {id} is used more than once"
            ))
        }
        None => Ok(()),
    }
}

/// The names of the classes a filter lets through, in class-id order.
fn class_names(filter: ClassFilter) -> Vec<String> {
    (0..NUM_CLASSES as u8)
        .filter(|&class| filter.contains(class))
        .map(|class| CLASS_NAMES[class as usize].to_string())
        .collect()
}

/// A filter with exactly the named classes; an unknown name is an error.
fn class_filter(names: &[String]) -> Result<ClassFilter, String> {
    let mut filter = ClassFilter::none();
    for name in names {
        let Some(class) = CLASS_NAMES.iter().position(|known| known == name) else {
            return Err(format!("unknown class \"{name}\""));
        };
        filter.set(class as u8, true);
    }
    Ok(filter)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A valid file with one line "door" (id 1) and one zone "desk" (id 2), with
    /// `edit` applied to its text first.
    fn file_with(edit: impl Fn(&str) -> String) -> String {
        let text = r#"{
            "version": 1,
            "lines": [{ "id": 1, "name": "door", "a": { "x": 0.5, "y": 0.1 },
                        "b": { "x": 0.5, "y": 0.9 }, "classes": ["person"] }],
            "zones": [{ "id": 2, "name": "desk",
                        "points": [{ "x": 0.1, "y": 0.6 }, { "x": 0.4, "y": 0.6 },
                                   { "x": 0.4, "y": 0.9 }],
                        "anchor": "center", "classes": ["person"], "dwell_s": 10.0 }]
        }"#;
        edit(text)
    }

    fn error_of(text: &str) -> String {
        Rules::from_json(text).expect_err("should be rejected")
    }

    #[test]
    fn events_rules_json_accepts_the_base_file() {
        let rules = Rules::from_json(&file_with(str::to_string)).unwrap();
        assert_eq!(
            rules.lines[0].anchor,
            Anchor::BottomCenter,
            "default anchor"
        );
        assert_eq!(rules.zones[0].anchor, Anchor::Center);
        assert_eq!(rules.name(2), Some("desk"));
    }

    #[test]
    fn events_rules_json_rejects() {
        let e = error_of(&file_with(|t| {
            t.replace("\"version\": 1", "\"version\": 2")
        }));
        assert!(e.contains("version 2"), "{e}");

        let e = error_of(&file_with(|t| t.replacen("0.9", "1.5", 1)));
        assert!(e.starts_with("line \"door\" (id 1):"), "{e}");
        assert!(e.contains("outside the frame"), "{e}");

        // serde_json reads 1e999 (infinity) as an error, so no NaN ever gets in; the
        // parser's message can't name the rule.
        let e = error_of(&file_with(|t| t.replacen("0.9", "1e999", 1)));
        assert!(e.starts_with("not a valid rules file"), "{e}");

        let zero_length =
            file_with(|t| t.replace("\"y\": 0.9 }, \"classes\"", "\"y\": 0.1 }, \"classes\""));
        let e = error_of(&zero_length);
        assert!(e.starts_with("line \"door\" (id 1):"), "{e}");
        assert!(e.contains("same point"), "{e}");

        let e = error_of(&file_with(|t| t.replace("{ \"x\": 0.4, \"y\": 0.6 },", "")));
        assert_eq!(e, "zone \"desk\" (id 2): needs at least 3 points, has 2");

        let e = error_of(&file_with(|t| {
            t.replacen("[\"person\"]", "[\"unicorn\"]", 1)
        }));
        assert!(e.starts_with("line \"door\" (id 1):"), "{e}");
        assert!(e.contains("unknown class \"unicorn\""), "{e}");

        let e = error_of(&file_with(|t| t.replace("\"id\": 2", "\"id\": 1")));
        assert!(e.contains("(id 1)"), "{e}");
        assert!(e.contains("used more than once"), "{e}");

        let e = error_of(&file_with(|t| t.replace("10.0", "0.5")));
        assert!(e.starts_with("zone \"desk\" (id 2):"), "{e}");
        assert!(e.contains("dwell_s 0.5"), "{e}");

        let e = error_of(&file_with(|t| t.replace("\"id\": 1", "\"id\": 0")));
        assert!(e.starts_with("line \"door\" (id 0):"), "{e}");
    }
}
