//! Copy as Markdown: the latency table as text, ready to paste into notes or a
//! report.
//!
//! The output is a header line naming the execution provider and the detect interval
//! (the two settings that change the numbers most), a blank line, then one Markdown
//! table row per table row:
//!
//! ```text
//! EP CoreML · detect every frame
//!
//! | Stage | last | p50 | p95 |
//! |---|--:|--:|--:|
//! | Sensor → app | 75.1 | 74.9 | 85.0 |
//! | **App total** | **80.2** | **79.8** | **94.0** |
//! ```
//!
//! `--:` right-aligns the number columns. The rows come from the same tables as the
//! page's grid (`rows.rs`), so the two can't drift apart. This
//! runs only when Copy is clicked, so it may allocate.

use std::fmt::Write;

use super::rows::{DETECT_ROWS, Row, VIDEO_ROWS, row_name};

/// The table's header and its alignment row.
const TABLE_HEAD: &str = "| Stage | last | p50 | p95 |\n|---|--:|--:|--:|\n";

/// The Markdown for one set of table cells (`DashText::latency`, or what Hold froze).
/// `ep` is the status bar's `EP CoreML` chip, `None` with `--no-detect`, which leaves
/// the detector rows out. `interval` is `DashText::detect_interval`. `partial` marks
/// an App total without the sensor stage, as the page does.
pub fn markdown(
    cells: &[[String; 3]; 10],
    ep: Option<&str>,
    interval: &str,
    partial: bool,
) -> String {
    let mut md = String::new();
    // Writing into a String can't fail, so the `fmt::Result`s are ignored.
    let _ = match ep {
        Some(ep) => writeln!(md, "{ep} · detect {interval}"),
        None => writeln!(md, "detect off"),
    };
    md.push('\n');
    md.push_str(TABLE_HEAD);
    let detector_rows: &[Row] = if ep.is_some() { &DETECT_ROWS } else { &[] };
    for row in VIDEO_ROWS.iter().chain(detector_rows) {
        table_row(&mut md, row, &cells[row.index], partial);
    }
    md
}

/// One `| name | last | p50 | p95 |` line. Totals are bold, as in the grid.
fn table_row(md: &mut String, row: &Row, cells: &[String; 3], partial: bool) {
    let name = row_name(row, partial);
    md.push('|');
    for text in std::iter::once(name).chain(cells.iter().map(|c| c.as_str())) {
        // DashText pads its cells for the monospace grid; Markdown needs no padding.
        let text = text.trim();
        let _ = if row.total {
            write!(md, " **{text}** |")
        } else {
            write!(md, " {text} |")
        };
    }
    md.push('\n');
}
