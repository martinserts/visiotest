//! The rule editor: drawing and editing lines and zones with the mouse.
//!
//! **An editor is a state machine.** The same input means different things at different
//! moments: a press starts a new line with the Line tool, adds a corner with the Zone
//! tool, and grabs whatever is under the pointer with the Select tool. So the editor
//! keeps a small state (the [`Tool`], what is being dragged, a half-drawn shape), and
//! each input ([`EditorInput`] from the pointer, [`EditorKey`] from the keyboard) moves it
//! to the next state:
//!
//! ```text
//! Select ──press on handle──► DraggingHandle ──release──► Select
//!   │  └──press on body────► DraggingBody ────release──► Select
//!   │
//! Line ──press──► DrawingLine ──release (long enough)──► Select (+ new line, selected)
//! Zone ──click──► DrawingZone ──click first corner / Enter──► Select (+ new zone)
//! ```
//!
//! After a new shape the tool goes back to Select, so the next click doesn't start
//! another one by accident. **Esc** ([`EditorKey::Cancel`]) unwinds one level at a
//! time: it drops a half-drawn shape, else clears the selection, else leaves edit mode
//! ([`Editor::is_editing`] turns false).
//!
//! **Two coordinate systems.** Rules are stored *normalised* (`0..=1` of the frame), but
//! the pointer lives in screen points. [`Viewport`] converts between them. Hit tests
//! ("is the pointer on this corner?") are done in **screen** space, so an 8-point grab
//! radius is 8 points on both axes and at every window size; in normalised units it
//! would be stretched 16:9.
//!
//! **No egui types.** The editor takes plain [`Point`]s and returns an [`Outcome`], and
//! the UI turns pointer events into inputs and draws what [`Editor::draft`] and
//! [`Editor::selected`] say. That keeps every rule above a unit test (see `tests.rs`).
//!
//! The editor never decides when to save or to send rules to the detector: it only
//! reports [`Outcome::changed`], and the caller acts on it.

mod hit;
mod moves;
#[cfg(test)]
mod tests;

use crate::events::geometry::{Point, to_pixels};
use crate::events::rules::{MAX_ZONE_POINTS, MIN_ZONE_POINTS, RuleId, Rules};

/// Grab radius around a line end or zone corner, in screen points: a comfortable target
/// for a mouse, a bit bigger than the 5-point handle that is drawn.
pub const HANDLE_RADIUS: f32 = 8.0;
/// How close to a line (screen points) counts as "on it", for moving the whole line. A
/// line is drawn only 3 points wide, so it needs a little slack to be grabbable.
pub const LINE_GRAB: f32 = 6.0;
/// A new line shorter than this (screen points) was almost certainly a click, not a
/// line, so it is discarded.
pub const MIN_LINE_PX: f32 = 12.0;

/// The on-screen video rect, in screen points: where normalised `(0, 0)` and `(1, 1)` are.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Viewport {
    /// The top-left corner of the video.
    pub origin: Point,
    pub width: f32,
    pub height: f32,
}

impl Viewport {
    /// Screen point → normalised frame coordinates, clamped to `0..=1`. A drag past the
    /// video's edge pins the point to the edge: a rule outside the frame could never fire.
    pub fn to_norm(self, screen: Point) -> Point {
        Point::new(
            fraction(screen.x - self.origin.x, self.width),
            fraction(screen.y - self.origin.y, self.height),
        )
    }

    /// Normalised frame coordinates → screen point.
    pub fn to_screen(self, norm: Point) -> Point {
        let offset = to_pixels(norm, self.width, self.height);
        Point::new(self.origin.x + offset.x, self.origin.y + offset.y)
    }
}

/// `offset / size` clamped to `0..=1`; 0 for an empty video rect (no frame yet), which
/// would otherwise divide by zero.
fn fraction(offset: f32, size: f32) -> f32 {
    if size <= 0.0 {
        return 0.0;
    }
    (offset / size).clamp(0.0, 1.0)
}

/// What the pointer does.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Tool {
    /// Grab, move and delete existing rules.
    #[default]
    Select,
    /// Press and drag to draw a line.
    Line,
    /// Click corner after corner to draw a zone.
    Zone,
}

/// What is under the pointer (see [`Editor::hit_test`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Hit {
    /// A line end (`index` 0 = `a`, 1 = `b`) or a zone corner (`index` into `points`).
    Handle { rule: RuleId, index: usize },
    /// Near a line, or inside a zone.
    Body { rule: RuleId },
}

/// One pointer event, positions in screen points.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum EditorInput {
    /// The pointer moved with no button down (the zone's rubber-band edge follows it).
    Hover(Point),
    /// The primary button went down.
    Press(Point),
    /// The pointer moved with the primary button down. `snap` (Shift) makes a new line
    /// horizontal or vertical.
    Drag { pos: Point, snap: bool },
    /// The primary button went up. A drag ends where the last `Drag` put it.
    Release,
    /// A right-click: on a zone corner, removes that corner.
    SecondaryClick(Point),
}

/// One key. None of them needs a position, so they don't need the [`Viewport`] either.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EditorKey {
    /// Esc: drop the draft, else deselect, else leave edit mode.
    Cancel,
    /// Enter: close the zone being drawn.
    Finish,
    /// Remove the selected rule. Both egui's `Key::Delete` and `Key::Backspace` map here
    /// (a Mac keyboard's "delete" key is Backspace).
    Delete,
}

/// What an input did, for the caller to act on.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Outcome {
    /// `rules` were changed (a rule added, moved or removed): save them and send them on.
    /// Hover, selection changes and drafts don't count.
    pub changed: bool,
}

impl Outcome {
    fn changed() -> Self {
        Self { changed: true }
    }
}

/// A shape being drawn, not yet in `Rules`. Normalised coordinates.
#[derive(Clone, Debug, PartialEq)]
pub enum Draft {
    /// A line being dragged out from `a` to `b`.
    Line { a: Point, b: Point },
    /// A zone's corners so far, and where the pointer is (the rubber-band edge's end).
    Zone {
        points: Vec<Point>,
        hover: Option<Point>,
    },
}

/// What the primary button is dragging.
#[derive(Clone, Debug, PartialEq)]
enum Drag {
    /// One line end or zone corner.
    Handle { rule: RuleId, index: usize },
    /// A whole shape. `grab` = where the pointer pressed, normalised; `start` = the
    /// shape's points then. Each drag moves the points from `start`, not from where the
    /// last drag left them, so rounding never accumulates into drift.
    Body {
        rule: RuleId,
        grab: Point,
        start: Vec<Point>,
    },
    /// The end `b` of the line draft.
    NewLine,
}

/// The editor's state: edit mode, the tool, the selection and what is being drawn or
/// dragged. `RulesSession` owns one, next to the `Rules` it edits. Every field is
/// private, so edit mode and the selection change only through the methods below.
#[derive(Debug, Default)]
pub struct Editor {
    /// Edit mode (key **E**, or the card's ✏): the video takes the mouse.
    editing: bool,
    tool: Tool,
    /// The rule the card highlights and Delete removes. Rules are removed through
    /// [`Editor::remove`], so it never names a removed rule; outside edit mode it is
    /// always `None`.
    selected: Option<RuleId>,
    drag: Option<Drag>,
    draft: Option<Draft>,
}

impl Editor {
    /// Not in edit mode, the Select tool, nothing selected.
    pub fn new() -> Self {
        Self::default()
    }

    /// In edit mode: the video takes the mouse and the rules show their handles.
    pub fn is_editing(&self) -> bool {
        self.editing
    }

    /// E or the card's ✏: enters edit mode, or leaves it (see `leave`).
    pub fn toggle_editing(&mut self) {
        if self.editing {
            self.leave();
        } else {
            self.editing = true;
        }
    }

    /// Leaves edit mode: back to the Select tool (dropping a half-drawn shape and any
    /// drag) with nothing selected, so the next edit mode starts clean.
    fn leave(&mut self) {
        self.editing = false;
        self.set_tool(Tool::Select);
        self.selected = None;
    }

    /// The current tool.
    pub fn tool(&self) -> Tool {
        self.tool
    }

    /// The selected rule, if any (always `None` outside edit mode).
    pub fn selected(&self) -> Option<RuleId> {
        self.selected
    }

    /// Selects rule `id` (a click on its row in the card). Only in edit mode: outside
    /// it nothing is ever selected, so the card and the video never highlight a rule.
    pub fn select(&mut self, id: RuleId) {
        if self.editing {
            self.selected = Some(id);
        }
    }

    /// Switches tools. A half-drawn shape and any drag are dropped; the selection stays.
    pub fn set_tool(&mut self, tool: Tool) {
        self.tool = tool;
        self.draft = None;
        self.drag = None;
    }

    /// Removes rule `id` (the card's 🗑 or Delete) and forgets it if it was selected or
    /// being dragged; false if there was no such rule.
    pub fn remove(&mut self, id: RuleId, rules: &mut Rules) -> bool {
        if self.selected == Some(id) {
            self.selected = None;
        }
        let dragged = match &self.drag {
            Some(Drag::Handle { rule, .. } | Drag::Body { rule, .. }) => Some(*rule),
            Some(Drag::NewLine) | None => None,
        };
        if dragged == Some(id) {
            self.drag = None;
        }
        rules.remove(id)
    }

    /// A press is moving a rule, a handle or a line draft. `RulesSession::tick` waits for
    /// the release before saving, so a drag writes the file once rather than four times
    /// a second.
    pub fn is_dragging(&self) -> bool {
        self.drag.is_some()
    }

    /// The shape being drawn, for the UI to show.
    pub fn draft(&self) -> Option<&Draft> {
        self.draft.as_ref()
    }

    /// What is under `screen`: handles before bodies, newest rule first.
    pub fn hit_test(rules: &Rules, screen: Point, view: &Viewport) -> Option<Hit> {
        hit::hit_test(rules, screen, view)
    }

    /// Feeds one pointer event through the state machine. It doesn't check edit mode:
    /// the UI feeds it only in edit mode, and the tests drive it directly.
    pub fn on_input(&mut self, input: EditorInput, rules: &mut Rules, view: &Viewport) -> Outcome {
        match input {
            EditorInput::Hover(pos) => self.hover(pos, view),
            EditorInput::Press(pos) => match self.tool {
                Tool::Select => self.press_select(pos, rules, view),
                Tool::Line => self.press_line(pos, view),
                Tool::Zone => self.press_zone(pos, rules, view),
            },
            EditorInput::Drag { pos, snap } => self.drag_to(pos, snap, rules, view),
            EditorInput::Release => self.release(rules, view),
            EditorInput::SecondaryClick(pos) => self.remove_corner(pos, rules, view),
        }
    }

    /// Feeds one key through the state machine. Like [`Editor::on_input`], it doesn't
    /// check edit mode: `App::keys` sends keys only in edit mode.
    pub fn on_key(&mut self, key: EditorKey, rules: &mut Rules) -> Outcome {
        match key {
            EditorKey::Cancel => self.cancel(),
            EditorKey::Finish => self.finish_zone(rules),
            EditorKey::Delete => self.delete_selected(rules),
        }
    }

    /// Moves the zone draft's rubber-band edge. Nothing else follows a hovering pointer.
    fn hover(&mut self, pos: Point, view: &Viewport) -> Outcome {
        if let Some(Draft::Zone { hover, .. }) = &mut self.draft {
            *hover = Some(view.to_norm(pos));
        }
        Outcome::default()
    }

    /// Select tool: grab a handle or a body, or deselect when the press hits nothing.
    fn press_select(&mut self, pos: Point, rules: &Rules, view: &Viewport) -> Outcome {
        match hit::hit_test(rules, pos, view) {
            Some(Hit::Handle { rule, index }) => {
                self.selected = Some(rule);
                self.drag = Some(Drag::Handle { rule, index });
            }
            Some(Hit::Body { rule }) => {
                self.selected = Some(rule);
                self.drag = Some(Drag::Body {
                    rule,
                    grab: view.to_norm(pos),
                    start: hit::shape_points(rules, rule),
                });
            }
            None => {
                self.selected = None;
                self.drag = None;
            }
        }
        Outcome::default()
    }

    /// Line tool: start a zero-length line draft; the drag pulls its end `b`.
    fn press_line(&mut self, pos: Point, view: &Viewport) -> Outcome {
        let a = view.to_norm(pos);
        self.draft = Some(Draft::Line { a, b: a });
        self.drag = Some(Drag::NewLine);
        Outcome::default()
    }

    /// Zone tool: close the zone on its first corner, otherwise add a corner. Reaching
    /// [`MAX_ZONE_POINTS`] closes it too.
    fn press_zone(&mut self, pos: Point, rules: &mut Rules, view: &Viewport) -> Outcome {
        let draft = self.draft.get_or_insert(Draft::Zone {
            points: Vec::new(),
            hover: None,
        });
        let Draft::Zone { points, .. } = draft else {
            // Only the Line tool makes a line draft, and switching tools drops it.
            return Outcome::default();
        };
        let on_first_corner = points
            .first()
            .is_some_and(|&first| distance(view.to_screen(first), pos) <= HANDLE_RADIUS);
        if on_first_corner && points.len() >= MIN_ZONE_POINTS {
            return self.close_zone(rules);
        }
        points.push(view.to_norm(pos));
        if points.len() >= MAX_ZONE_POINTS {
            return self.close_zone(rules);
        }
        Outcome::default()
    }

    /// Enter: close a zone draft that already has enough corners; otherwise nothing.
    fn finish_zone(&mut self, rules: &mut Rules) -> Outcome {
        match &self.draft {
            Some(Draft::Zone { points, .. }) if points.len() >= MIN_ZONE_POINTS => {
                self.close_zone(rules)
            }
            _ => Outcome::default(),
        }
    }

    /// Turns the zone draft into a rule, selects it and goes back to Select.
    fn close_zone(&mut self, rules: &mut Rules) -> Outcome {
        let Some(Draft::Zone { points, .. }) = self.draft.take() else {
            return Outcome::default();
        };
        // `Rules::add_zone` expects at least three corners; both callers checked.
        debug_assert!(points.len() >= MIN_ZONE_POINTS);
        self.selected = Some(rules.add_zone(points));
        self.tool = Tool::Select;
        Outcome::changed()
    }

    /// Moves whatever the press grabbed.
    fn drag_to(&mut self, pos: Point, snap: bool, rules: &mut Rules, view: &Viewport) -> Outcome {
        match &self.drag {
            Some(Drag::NewLine) => {
                self.pull_line_draft(pos, snap, view);
                Outcome::default()
            }
            Some(Drag::Handle { rule, index }) => {
                moves::move_handle(rules, *rule, *index, view.to_norm(pos))
            }
            Some(Drag::Body { rule, grab, start }) => {
                let offset = moves::clamped_offset(start, view.to_norm(pos), *grab);
                moves::move_body(rules, *rule, start, offset)
            }
            None => Outcome::default(),
        }
    }

    /// Puts the line draft's end at `pos`. With `snap`, the line becomes horizontal or
    /// vertical, whichever is closer: the smaller of |dx| and |dy| is zeroed. That is done
    /// in screen space, so "horizontal" means horizontal on screen, then converted.
    fn pull_line_draft(&mut self, pos: Point, snap: bool, view: &Viewport) {
        let Some(Draft::Line { a, b }) = &mut self.draft else {
            return;
        };
        let start = view.to_screen(*a);
        let mut end = pos;
        if snap {
            if (end.x - start.x).abs() >= (end.y - start.y).abs() {
                end.y = start.y;
            } else {
                end.x = start.x;
            }
        }
        *b = view.to_norm(end);
    }

    /// Ends any drag. A line draft becomes a rule if it is long enough on screen.
    fn release(&mut self, rules: &mut Rules, view: &Viewport) -> Outcome {
        let Some(Drag::NewLine) = self.drag.take() else {
            return Outcome::default();
        };
        let Some(Draft::Line { a, b }) = self.draft.take() else {
            return Outcome::default();
        };
        if distance(view.to_screen(a), view.to_screen(b)) < MIN_LINE_PX {
            // A click, not a line. The Line tool stays, ready for a real drag.
            return Outcome::default();
        }
        self.selected = Some(rules.add_line(a, b));
        self.tool = Tool::Select;
        Outcome::changed()
    }

    /// Right-click on a zone corner removes it, as long as the zone keeps at least
    /// [`MIN_ZONE_POINTS`]. Only with the Select tool, so a right-click while drawing does
    /// nothing surprising.
    fn remove_corner(&mut self, pos: Point, rules: &mut Rules, view: &Viewport) -> Outcome {
        if self.tool != Tool::Select {
            return Outcome::default();
        }
        let Some(Hit::Handle { rule, index }) = hit::hit_test(rules, pos, view) else {
            return Outcome::default();
        };
        // A line end is a handle too; `zone_mut` misses for lines.
        let Some(zone) = rules.zone_mut(rule) else {
            return Outcome::default();
        };
        if zone.points.len() <= MIN_ZONE_POINTS {
            return Outcome::default();
        }
        zone.points.remove(index);
        Outcome::changed()
    }

    /// Esc, one level at a time: the draft, else the selection, else edit mode itself
    /// (see `leave`).
    fn cancel(&mut self) -> Outcome {
        if self.draft.is_some() {
            self.draft = None;
            self.drag = None;
            return Outcome::default();
        }
        if self.selected.is_some() {
            self.selected = None;
            self.drag = None;
            return Outcome::default();
        }
        self.leave();
        Outcome::default()
    }

    /// Delete/Backspace: removes the selected rule.
    fn delete_selected(&mut self, rules: &mut Rules) -> Outcome {
        let Some(id) = self.selected else {
            return Outcome::default();
        };
        if self.remove(id, rules) {
            Outcome::changed()
        } else {
            Outcome::default()
        }
    }
}

/// Euclidean distance between two points.
fn distance(p: Point, q: Point) -> f32 {
    (p.x - q.x).hypot(p.y - q.y)
}
