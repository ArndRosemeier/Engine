//! The engine's version number and its changelog ("What's new").
//!
//! The single source is `CHANGELOG.md` at the repository root: the newest
//! entry is the current version. Every merge into main adds one entry and
//! raises the version by 0.001.

/// The whole changelog, newest version first.
pub const CHANGELOG: &str = include_str!("../../CHANGELOG.md");

/// One version and what it brought.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Release {
    /// For example "0.001".
    pub version: &'static str,
    /// Merge date, for example "2026-11-07" (empty if the entry has none).
    pub date: &'static str,
    /// The notes in Markdown, without the heading.
    pub notes: &'static str,
}

impl Release {
    /// The version as a build count: "0.001" is 1, "1.234" is 1234.
    pub fn build(&self) -> u32 {
        parse_build(self.version).unwrap_or(0)
    }
}

/// The current engine version, for example "0.001".
pub fn version() -> &'static str {
    releases().first().map_or("0.000", |r| r.version)
}

/// Every version, newest first.
pub fn releases() -> Vec<Release> {
    parse(CHANGELOG)
}

/// Turn "0.001" into 1. `None` if it is not `<whole>.<three digits>`.
pub fn parse_build(version: &str) -> Option<u32> {
    let (whole, frac) = version.split_once('.')?;
    if frac.len() != 3 || whole.is_empty() {
        return None;
    }
    let whole: u32 = whole.parse().ok()?;
    let frac: u32 = frac.parse().ok()?;
    Some(whole * 1000 + frac)
}

/// Turn a build count back into a version string: 1 is "0.001".
pub fn format_build(build: u32) -> String {
    format!("{}.{:03}", build / 1000, build % 1000)
}

fn parse(text: &'static str) -> Vec<Release> {
    let mut out = Vec::new();
    // Byte offsets of every "## " heading line.
    let mut starts = Vec::new();
    let mut offset = 0;
    for line in text.split_inclusive('\n') {
        if line.starts_with("## ") {
            starts.push(offset);
        }
        offset += line.len();
    }
    for (i, &start) in starts.iter().enumerate() {
        let end = starts.get(i + 1).copied().unwrap_or(text.len());
        let block = &text[start..end];
        let (heading, notes) = block.split_once('\n').unwrap_or((block, ""));
        let heading = heading[3..].trim();
        let (version, rest) = heading.split_once(' ').unwrap_or((heading, ""));
        let date = rest.trim().trim_start_matches('(').trim_end_matches(')');
        out.push(Release {
            version,
            date,
            notes: notes.trim(),
        });
    }
    out
}

/// Draw the "What's new" button in the bottom right corner and, once
/// clicked, a window with the changelog. The engine calls this every frame
/// unless [`crate::World::set_whats_new_button`] turned it off; call it
/// yourself to show it in your own UI.
pub fn whats_new_ui(ctx: &egui::Context) {
    let mut open = ctx.data(|d| d.get_temp::<bool>(open_id()).unwrap_or(false));

    egui::Area::new(egui::Id::new("engine_whats_new_button"))
        .anchor(egui::Align2::RIGHT_BOTTOM, [-12.0, -12.0])
        .order(egui::Order::Foreground)
        .show(ctx, |ui| {
            let label = format!("Engine {}  ·  What's new", version());
            if ui.small_button(label).clicked() {
                open = !open;
            }
        });

    if open {
        egui::Window::new(format!("What's new in Engine {}", version()))
            .id(egui::Id::new("engine_whats_new_window"))
            .open(&mut open)
            .collapsible(false)
            .default_width(460.0)
            .anchor(egui::Align2::RIGHT_BOTTOM, [-12.0, -40.0])
            .show(ctx, |ui| {
                egui::ScrollArea::vertical()
                    .max_height(420.0)
                    .show(ui, |ui| {
                        for release in releases() {
                            ui.heading(format!("{}  ({})", release.version, release.date));
                            for line in note_lines(release.notes) {
                                ui.label(line);
                            }
                            ui.add_space(8.0);
                        }
                    });
            });
    }
    set_whats_new_open(ctx, open);
}

/// Open or close the "What's new" window from code.
pub fn set_whats_new_open(ctx: &egui::Context, open: bool) {
    ctx.data_mut(|d| d.insert_temp(open_id(), open));
}

fn open_id() -> egui::Id {
    egui::Id::new("engine_whats_new_open")
}

/// Join wrapped Markdown bullet lines into one string per bullet.
fn note_lines(notes: &str) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    for line in notes.lines() {
        let t = line.trim().replace('`', "");
        let t = t.as_str();
        if t.is_empty() {
            continue;
        }
        if let Some(item) = t.strip_prefix("- ").or_else(|| t.strip_prefix("* ")) {
            out.push(format!("• {item}"));
        } else if let Some(last) = out.last_mut() {
            last.push(' ');
            last.push_str(t);
        } else {
            out.push(t.to_string());
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn changelog_counts_up_by_one_thousandth_from_0_001() {
        let releases = releases();
        assert!(
            !releases.is_empty(),
            "CHANGELOG.md has no '## <version>' entry"
        );
        let newest = releases[0].build();
        assert!(newest >= 1, "bad top version {:?}", releases[0].version);
        for (i, r) in releases.iter().enumerate() {
            let expected = newest - i as u32;
            assert_eq!(
                parse_build(r.version),
                Some(expected),
                "CHANGELOG.md entry {i} is {:?}, expected {} (newest first, one entry per 0.001)",
                r.version,
                format_build(expected)
            );
            assert!(!r.notes.is_empty(), "version {} has no notes", r.version);
            assert_eq!(
                r.date.len(),
                10,
                "version {} needs a (YYYY-MM-DD) date",
                r.version
            );
        }
        assert_eq!(releases.last().unwrap().version, "0.001");
        assert_eq!(version(), releases[0].version);
    }

    #[test]
    fn build_numbers_round_trip() {
        assert_eq!(parse_build("0.001"), Some(1));
        assert_eq!(parse_build("0.010"), Some(10));
        assert_eq!(parse_build("1.000"), Some(1000));
        assert_eq!(parse_build("0.01"), None);
        assert_eq!(format_build(1), "0.001");
        assert_eq!(format_build(1234), "1.234");
    }

    #[test]
    fn notes_join_wrapped_bullets() {
        let lines = note_lines("- one\n  two\n- three");
        assert_eq!(lines, vec!["• one two".to_string(), "• three".to_string()]);
    }
}
