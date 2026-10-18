//! The rules being edited: the working copy, the rule editor, and when to save them.
//!
//! The editor (on the video) and the Events card change [`RulesSession::rules`] in
//! place, and every change is reported with [`RulesSession::changed`]. That hands the
//! detector its own copy at once and marks the rules dirty. Saving follows a small
//! policy, so the file is written often enough to never lose an edit, but not on
//! every mouse move:
//!
//! - dirty after every change;
//! - saved on the 4 Hz tick ([`RulesSession::tick`]), but never in the middle of a
//!   drag, so dragging a zone writes the file once, when it is let go;
//! - a failed save (a read-only file, a missing folder) is shown, and not retried until
//!   the next change, so a broken path isn't hammered four times a second;
//! - once more on exit ([`RulesSession::flush`]), so an edit made just before closing
//!   isn't lost.
//!
//! No egui and no threads: `App` calls it, and the tests drive it directly.

use std::path::{Path, PathBuf};

use crate::detect::worker::DetectorSettings;
use crate::events::rules::Rules;
use crate::ui::rule_editor::Editor;

/// Whether the working copy of the rules is on disk.
#[derive(Clone, Debug, Default, PartialEq)]
pub enum SaveStatus {
    /// Nothing changed since startup.
    #[default]
    Clean,
    /// The last change was saved.
    Saved,
    /// The last save failed, with the error. The next change tries again.
    Failed(String),
}

impl SaveStatus {
    /// The last save failed: the card shows it as a warning.
    pub fn is_failed(&self) -> bool {
        matches!(self, SaveStatus::Failed(_))
    }
}

/// The working copy of the rules, the editor that changes it, and its save state.
pub struct RulesSession {
    /// The working copy; the detector has its own (`DetectorSettings::set_rules`).
    pub rules: Rules,
    /// The rule editor (edit mode, tool, selection, half-drawn shape).
    pub editor: Editor,
    /// The rules file (`--rules`).
    path: PathBuf,
    /// Changed since the last save attempt.
    dirty: bool,
    status: SaveStatus,
}

impl RulesSession {
    /// `rules` as loaded from `path` (empty if the file didn't exist): clean, edit mode
    /// off. The caller hands the detector its first copy; loading isn't a change, so an
    /// unchanged file isn't rewritten on the first tick.
    pub fn new(rules: Rules, path: PathBuf) -> Self {
        Self {
            rules,
            editor: Editor::new(),
            path,
            dirty: false,
            status: SaveStatus::Clean,
        }
    }

    /// The rules changed (editor or card): the detector adopts them on its next run,
    /// and the next `tick` saves them.
    pub fn changed(&mut self, detector: &DetectorSettings) {
        detector.set_rules(&self.rules);
        self.dirty = true;
    }

    /// The 4 Hz tick: saves if dirty and no drag is in progress. A drag only postpones
    /// the save: the first tick after the release writes it.
    pub fn tick(&mut self) {
        if self.dirty && !self.editor.is_dragging() {
            self.save();
        }
    }

    /// Saves if dirty (on exit). Returns the error text if that save failed; `None`
    /// when it worked or there was nothing to save (the file isn't touched then).
    pub fn flush(&mut self) -> Option<&str> {
        if !self.dirty {
            return None;
        }
        self.save();
        match &self.status {
            SaveStatus::Failed(e) => Some(e),
            _ => None,
        }
    }

    /// Whether the rules are on disk, for the card's save line.
    pub fn status(&self) -> &SaveStatus {
        &self.status
    }

    /// The rules file, for the card's save line.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// One save attempt; clears `dirty` either way, which is what stops a failure
    /// from being retried before the next change.
    fn save(&mut self) {
        self.dirty = false;
        self.status = match self.rules.save(&self.path) {
            Ok(()) => SaveStatus::Saved,
            // `:#` prints anyhow's whole chain ("writing …: No such file or directory").
            Err(e) => SaveStatus::Failed(format!("{e:#}")),
        };
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::events::geometry::Point;
    use crate::ui::rule_editor::{EditorInput, Viewport};

    /// The rule editor tests' viewport: half of a 1920×1080 frame, next to a panel.
    const VIEW: Viewport = Viewport {
        origin: Point { x: 100.0, y: 50.0 },
        width: 960.0,
        height: 540.0,
    };

    /// A unique path under the system temp dir, so parallel tests never collide.
    fn temp_path(test: &str) -> PathBuf {
        std::env::temp_dir().join(format!(
            "visiotest-session-{}-{test}.json",
            std::process::id()
        ))
    }

    /// A rules file inside a directory that doesn't exist (yet), so saving fails.
    fn missing_dir(test: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "visiotest-session-{}-{test}-missing",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        dir
    }

    fn remove_with_tmp(path: &Path) {
        let _ = std::fs::remove_file(path);
        let _ = std::fs::remove_file(path.with_extension("json.tmp"));
    }

    /// A session with one vertical line through the middle, saving to `path`.
    fn one_line(path: PathBuf) -> RulesSession {
        let mut rules = Rules::default();
        rules.add_line(Point::new(0.5, 0.2), Point::new(0.5, 0.8));
        RulesSession::new(rules, path)
    }

    #[test]
    fn rules_session_changed_then_tick_saves_once() {
        let path = temp_path("tick_saves_once");
        remove_with_tmp(&path);
        let detector = DetectorSettings::default();
        let mut session = one_line(path.clone());

        session.changed(&detector);
        session.tick();
        let loaded = Rules::load(&path);
        let status = session.status().clone();
        // Not dirty any more, so a second tick leaves the (removed) file alone.
        remove_with_tmp(&path);
        session.tick();
        let rewritten = path.exists();
        remove_with_tmp(&path);

        assert_eq!(loaded.unwrap(), Some(session.rules.clone()));
        assert_eq!(status, SaveStatus::Saved);
        assert!(!rewritten);
    }

    #[test]
    fn rules_session_waits_for_drag_to_end() {
        let path = temp_path("waits_for_drag");
        remove_with_tmp(&path);
        let detector = DetectorSettings::default();
        let mut session = one_line(path.clone());
        let a = session.rules.lines[0].a;

        // Grab the line's `a` handle: a drag is in progress until the release.
        let s = &mut session;
        s.editor
            .on_input(EditorInput::Press(VIEW.to_screen(a)), &mut s.rules, &VIEW);
        let dragging = s.editor.is_dragging();
        s.changed(&detector);
        s.tick();
        let saved_mid_drag = path.exists();
        s.editor.on_input(EditorInput::Release, &mut s.rules, &VIEW);
        s.tick();
        let saved_after = path.exists();
        remove_with_tmp(&path);

        assert!(dragging);
        assert!(!saved_mid_drag);
        assert!(saved_after);
    }

    #[test]
    fn rules_session_failed_save_not_retried() {
        let dir = missing_dir("not_retried");
        let path = dir.join("rules.json");
        let detector = DetectorSettings::default();
        let mut session = one_line(path.clone());

        session.changed(&detector);
        session.tick();
        let failed = session.status().is_failed();
        // The folder appears, but nothing changed since the failure: no retry.
        std::fs::create_dir_all(&dir).unwrap();
        session.tick();
        let retried = path.exists();
        // The next change tries again, and now it works.
        session.changed(&detector);
        session.tick();
        let saved = path.exists();
        let status = session.status().clone();
        let _ = std::fs::remove_dir_all(&dir);

        assert!(failed);
        assert!(!retried);
        assert!(saved);
        assert_eq!(status, SaveStatus::Saved);
    }

    #[test]
    fn rules_session_flush_only_when_dirty() {
        let path = temp_path("flush_only_when_dirty");
        remove_with_tmp(&path);
        let detector = DetectorSettings::default();
        let mut session = one_line(path.clone());

        let clean_flush = session.flush().map(str::to_owned);
        let touched = path.exists();
        let clean_status = session.status().clone();
        session.changed(&detector);
        let dirty_flush = session.flush().map(str::to_owned);
        let written = path.exists();
        remove_with_tmp(&path);

        let dir = missing_dir("flush_fails");
        let mut failing = one_line(dir.join("rules.json"));
        failing.changed(&detector);
        let error = failing.flush().map(str::to_owned);
        let _ = std::fs::remove_dir_all(&dir);

        assert_eq!(clean_flush, None);
        assert!(!touched);
        assert_eq!(clean_status, SaveStatus::Clean);
        assert_eq!(dirty_flush, None);
        assert!(written);
        assert!(error.is_some_and(|e| !e.is_empty()));
    }

    #[test]
    fn rules_session_changed_hands_rules_to_detector() {
        let detector = DetectorSettings::default();
        let mut session = one_line(temp_path("hands_rules_to_detector"));

        session.changed(&detector);

        let (version, rules) = detector.rules_if_newer(0).unwrap();
        assert!(version >= 1);
        assert_eq!(rules, session.rules);
    }
}
