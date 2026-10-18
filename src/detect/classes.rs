//! The 80 COCO classes, their 12 supercategory groups, and a filter over them.
//!
//! YOLO models trained on COCO predict one score per class, and the class id is simply
//! the index into [`CLASS_NAMES`]. For the UI we also want coarser groups ("Vehicles",
//! "Food", ...), so the filter can switch a whole group on or off at once.

use std::ops::Range;

use super::{Detection, NUM_CLASSES};

/// Class names in YOLO/COCO order, with the exact Ultralytics spellings.
/// Index = class id = position of that class's score in the model output.
// Kept on a few lines per group (rustfmt would put all 80 on their own lines).
#[rustfmt::skip]
pub const CLASS_NAMES: [&str; NUM_CLASSES] = [
    "person",
    // Vehicles (1..9)
    "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    // Outdoor (9..14)
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    // Animals (14..24)
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
    // Accessories (24..29)
    "backpack", "umbrella", "handbag", "tie", "suitcase",
    // Sports (29..39)
    "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket",
    // Kitchen (39..46)
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl",
    // Food (46..56)
    "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut",
    "cake",
    // Furniture (56..62)
    "chair", "couch", "potted plant", "bed", "dining table", "toilet",
    // Electronics (62..68)
    "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    // Appliances (68..73)
    "microwave", "oven", "toaster", "sink", "refrigerator",
    // Indoor (73..80)
    "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
];

/// A COCO supercategory.
///
/// These are COCO's own supercategories (from its annotation file), renamed to plural
/// UI labels. In the 80-class order the ids of each group happen to be contiguous, so a
/// group is just a `Range<u8>` and needs no per-class lookup table.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Group {
    People,
    Vehicles,
    Outdoor,
    Animals,
    Accessories,
    Sports,
    Kitchen,
    Food,
    Furniture,
    Electronics,
    Appliances,
    Indoor,
}

impl Group {
    /// Every group, in class-id order.
    pub const ALL: [Group; 12] = [
        Group::People,
        Group::Vehicles,
        Group::Outdoor,
        Group::Animals,
        Group::Accessories,
        Group::Sports,
        Group::Kitchen,
        Group::Food,
        Group::Furniture,
        Group::Electronics,
        Group::Appliances,
        Group::Indoor,
    ];

    /// Label shown in the UI.
    pub fn name(self) -> &'static str {
        match self {
            Group::People => "People",
            Group::Vehicles => "Vehicles",
            Group::Outdoor => "Outdoor",
            Group::Animals => "Animals",
            Group::Accessories => "Accessories",
            Group::Sports => "Sports",
            Group::Kitchen => "Kitchen",
            Group::Food => "Food",
            Group::Furniture => "Furniture",
            Group::Electronics => "Electronics",
            Group::Appliances => "Appliances",
            Group::Indoor => "Indoor",
        }
    }

    /// The class ids in this group (contiguous in COCO order).
    pub fn classes(self) -> Range<u8> {
        match self {
            Group::People => 0..1,
            Group::Vehicles => 1..9,
            Group::Outdoor => 9..14,
            Group::Animals => 14..24,
            Group::Accessories => 24..29,
            Group::Sports => 29..39,
            Group::Kitchen => 39..46,
            Group::Food => 46..56,
            Group::Furniture => 56..62,
            Group::Electronics => 62..68,
            Group::Appliances => 68..73,
            Group::Indoor => 73..80,
        }
    }

    /// The group a class id belongs to. Panics for an id ≥ 80, which the model can't produce.
    pub fn of(class: u8) -> Group {
        Group::ALL
            .into_iter()
            .find(|g| g.classes().contains(&class))
            .expect("COCO class id must be < 80")
    }
}

/// How many of a group's classes a filter shows; the UI draws it as a tri-state
/// checkbox (checked, indeterminate, empty).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GroupState {
    All,
    Some,
    None,
}

/// Which classes are shown, as a bitset: bit `c` set means class `c` is shown.
/// 80 classes fit in one `u128`, so the filter is a plain `Copy` value with no allocation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ClassFilter(u128);

impl Default for ClassFilter {
    fn default() -> Self {
        Self::all()
    }
}

impl ClassFilter {
    /// Every class shown.
    pub fn all() -> Self {
        Self(bits(0..NUM_CLASSES as u8))
    }

    /// Nothing shown.
    pub fn none() -> Self {
        Self(0)
    }

    /// Only the classes of one group shown.
    pub fn only(g: Group) -> Self {
        Self(bits(g.classes()))
    }

    pub fn contains(self, class: u8) -> bool {
        self.0 & (1u128 << class) != 0
    }

    pub fn set(&mut self, class: u8, on: bool) {
        self.set_bits(1u128 << class, on);
    }

    /// Shows or hides every class of a group.
    pub fn set_group(&mut self, g: Group, on: bool) {
        self.set_bits(bits(g.classes()), on);
    }

    pub fn group_state(self, g: Group) -> GroupState {
        let mask = bits(g.classes());
        match self.0 & mask {
            0 => GroupState::None,
            shown if shown == mask => GroupState::All,
            _ => GroupState::Some,
        }
    }

    fn set_bits(&mut self, mask: u128, on: bool) {
        if on {
            self.0 |= mask;
        } else {
            self.0 &= !mask;
        }
    }
}

/// How many detections of each class are in the newest result: the panel's live counts.
///
/// Counted over **all** classes, filtered or not, so the panel can also show that a
/// hidden class is present, but only over boxes the confidence slider lets through.
/// A fixed array, so counting never allocates.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ClassCounts([u16; NUM_CLASSES]);

impl Default for ClassCounts {
    fn default() -> Self {
        Self([0; NUM_CLASSES])
    }
}

impl ClassCounts {
    /// Counts the detections with `score ≥ min_score`. The detector keeps boxes down
    /// to `LOW_CONF` for the tracker, so the UI passes its slider value here to count
    /// what it draws.
    pub fn from_detections(dets: &[Detection], min_score: f32) -> Self {
        let shown = dets.iter().filter(|d| d.score >= min_score);
        Self::from_classes(shown.map(|d| d.class))
    }

    /// Counts from a list of class ids (e.g. the classes of the tracked objects).
    pub fn from_classes(classes: impl IntoIterator<Item = u8>) -> Self {
        let mut counts = Self::default();
        for class in classes {
            let n = &mut counts.0[usize::from(class)];
            *n = n.saturating_add(1);
        }
        counts
    }

    /// Objects of one class.
    pub fn class(&self, class: u8) -> u16 {
        self.0[usize::from(class)]
    }

    /// Objects of every class in a group (`u32`, so a sum of `u16`s can't overflow).
    pub fn group(&self, g: Group) -> u32 {
        g.classes().map(|c| u32::from(self.class(c))).sum()
    }
}

/// A mask with bits `r.start..r.end` set: all ones below `end`, minus all ones below `start`.
fn bits(r: Range<u8>) -> u128 {
    let below = |n: u8| (1u128 << n) - 1;
    below(r.end) & !below(r.start)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classes_names() {
        assert_eq!(CLASS_NAMES.len(), 80);
        assert_eq!(CLASS_NAMES[0], "person");
        assert_eq!(CLASS_NAMES[79], "toothbrush");
        assert_eq!(CLASS_NAMES[62], "tv");
        assert_eq!(CLASS_NAMES[67], "cell phone");
    }

    #[test]
    fn classes_groups_tile_0_to_80() {
        let mut next = 0;
        for g in Group::ALL {
            let r = g.classes();
            assert_eq!(
                r.start, next,
                "{g:?} must start where the previous group ended"
            );
            assert!(r.end > r.start, "{g:?} is empty");
            next = r.end;
        }
        assert_eq!(usize::from(next), NUM_CLASSES);
    }

    #[test]
    fn classes_group_of_agrees_with_ranges() {
        for g in Group::ALL {
            for c in g.classes() {
                assert_eq!(
                    Group::of(c),
                    g,
                    "class {c} ({})",
                    CLASS_NAMES[usize::from(c)]
                );
            }
        }
        assert_eq!(Group::of(0), Group::People);
        assert_eq!(Group::of(41), Group::Kitchen); // cup
    }

    #[test]
    fn classes_filter_all_none_only() {
        let all = ClassFilter::all();
        let none = ClassFilter::none();
        let people = ClassFilter::only(Group::People);
        assert_eq!(ClassFilter::default(), all);
        for c in 0..80 {
            assert!(all.contains(c));
            assert!(!none.contains(c));
            assert_eq!(people.contains(c), c == 0);
        }
    }

    #[test]
    fn classes_filter_set_and_group_state() {
        let mut f = ClassFilter::none();
        assert_eq!(f.group_state(Group::Kitchen), GroupState::None);
        f.set(41, true); // cup
        assert!(f.contains(41));
        assert_eq!(f.group_state(Group::Kitchen), GroupState::Some);
        f.set_group(Group::Kitchen, true);
        assert_eq!(f.group_state(Group::Kitchen), GroupState::All);
        assert_eq!(f.group_state(Group::Food), GroupState::None);
        f.set(41, false);
        assert_eq!(f.group_state(Group::Kitchen), GroupState::Some);
        f.set_group(Group::Kitchen, false);
        assert_eq!(f, ClassFilter::none());
    }

    fn det(class: u8) -> Detection {
        scored(class, 0.9)
    }

    fn scored(class: u8, score: f32) -> Detection {
        Detection {
            bbox: crate::detect::BBox::new(0.0, 0.0, 1.0, 1.0),
            class,
            score,
        }
    }

    #[test]
    fn class_counts_empty_is_zero() {
        let c = ClassCounts::from_detections(&[], 0.35);
        for g in Group::ALL {
            assert_eq!(c.group(g), 0, "{g:?}");
        }
        assert!((0..80).all(|k| c.class(k) == 0));
    }

    #[test]
    fn class_counts_people_and_a_cup() {
        let c = ClassCounts::from_detections(&[det(0), det(41), det(0), det(0)], 0.35);
        assert_eq!(c.class(0), 3);
        assert_eq!(c.class(41), 1); // cup
        assert_eq!(c.group(Group::People), 3);
        assert_eq!(c.group(Group::Kitchen), 1);
        assert_eq!(c.group(Group::Food), 0);
    }

    #[test]
    fn class_counts_skip_boxes_below_min_score() {
        // A 0.2 box is kept for the tracker's second round, but not drawn at 0.35.
        let c = ClassCounts::from_detections(&[scored(0, 0.2), scored(0, 0.35)], 0.35);
        assert_eq!(c.class(0), 1);
    }

    #[test]
    fn class_counts_from_classes() {
        // Three tracked people and a dog (class 16).
        let c = ClassCounts::from_classes([0, 16, 0, 0]);
        assert_eq!(c.class(0), 3);
        assert_eq!(c.class(16), 1);
        assert_eq!(c.group(Group::People), 3);
        assert_eq!(c.group(Group::Animals), 1);
        assert_eq!(ClassCounts::from_classes([]), ClassCounts::default());
    }
}
