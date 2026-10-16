//! Minimal uncompressed PDF writers for tests (the Rust side of upstream's
//! `conftest.build_pdf` and `_layout_pdf`).
#![allow(dead_code)]

pub const HELVETICA: &str = "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>";

/// A page: one line of text, positioned `(x, y, size, text)` lines, or a raw content stream.
pub enum PageSpec {
    Line(String),
    Lines(Vec<(f32, f32, f32, String)>),
    Raw(Vec<u8>),
}

fn escape(text: &str) -> String {
    text.replace('\\', r"\\").replace('(', r"\(").replace(')', r"\)")
}

/// A Helvetica-like font dictionary with explicit widths for codes 32..=126.
pub fn helvetica_with_widths(widths: &[u32], encoding: &str) -> String {
    let widths: Vec<String> = widths.iter().map(u32::to_string).collect();
    format!(
        "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding {encoding} /FirstChar 32 /LastChar 126 /Widths [{}] >>",
        widths.join(" ")
    )
}

fn assemble(objects: &[String]) -> Vec<u8> {
    let mut out = b"%PDF-1.4\n".to_vec();
    let mut offsets = Vec::new();
    for (i, body) in objects.iter().enumerate() {
        offsets.push(out.len());
        out.extend_from_slice(format!("{} 0 obj\n{body}\nendobj\n", i + 1).as_bytes());
    }
    let xref = out.len();
    out.extend_from_slice(format!("xref\n0 {}\n0000000000 65535 f \n", objects.len() + 1).as_bytes());
    for offset in offsets {
        out.extend_from_slice(format!("{offset:010} 00000 n \n").as_bytes());
    }
    out.extend_from_slice(
        format!("trailer\n<< /Size {} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n", objects.len() + 1).as_bytes(),
    );
    out
}

/// One page per spec on a 612x792 MediaBox, all in `font`.
pub fn build_pdf(pages: &[PageSpec], font: &str) -> Vec<u8> {
    build_pdf_with_outline(pages, font, &[], (612.0, 792.0))
}

/// [`build_pdf`] plus an outline of `(title, level, page)` entries (1-based).
pub fn build_pdf_with_outline(
    pages: &[PageSpec],
    font: &str,
    outline: &[(&str, usize, usize)],
    size: (f32, f32),
) -> Vec<u8> {
    let n = pages.len();
    let font_obj = 3 + 2 * n;
    let outline_root = font_obj + 1;
    let kids: Vec<String> = (0..n).map(|i| format!("{} 0 R", 3 + i)).collect();
    let mut objects = Vec::new();
    let outlines = if outline.is_empty() { String::new() } else { format!(" /Outlines {outline_root} 0 R") };
    objects.push(format!("<< /Type /Catalog /Pages 2 0 R{outlines} >>"));
    objects.push(format!("<< /Type /Pages /Kids [{}] /Count {n} >>", kids.join(" ")));
    for i in 0..n {
        objects.push(format!(
            "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {} {}] /Resources << /Font << /F1 {font_obj} 0 R >> >> /Contents {} 0 R >>",
            size.0,
            size.1,
            3 + n + i
        ));
    }
    for page in pages {
        let stream: Vec<u8> = match page {
            PageSpec::Raw(bytes) => bytes.clone(),
            PageSpec::Line(text) => format!("BT /F1 12 Tf 72 720 Td ({}) Tj ET", escape(text)).into_bytes(),
            PageSpec::Lines(lines) => lines
                .iter()
                .map(|(x, y, size, text)| format!("BT /F1 {size} Tf {x} {y} Td ({}) Tj ET", escape(text)))
                .collect::<Vec<_>>()
                .join(" ")
                .into_bytes(),
        };
        objects.push(format!(
            "<< /Length {} >>\nstream\n{}\nendstream",
            stream.len(),
            String::from_utf8_lossy(&stream)
        ));
    }
    objects.push(font.to_owned());
    if !outline.is_empty() {
        // Items are numbered after the root; parent/sibling links by level.
        let first_item = outline_root + 1;
        let mut parents: Vec<Option<usize>> = Vec::new();
        let mut stack: Vec<(usize, usize)> = Vec::new();
        for (i, (_, level, _)) in outline.iter().enumerate() {
            while stack.last().is_some_and(|&(l, _)| l >= *level) {
                stack.pop();
            }
            parents.push(stack.last().map(|&(_, idx)| idx));
            stack.push((*level, i));
        }
        let children = |p: Option<usize>| -> Vec<usize> { (0..outline.len()).filter(|&i| parents[i] == p).collect() };
        let top = children(None);
        objects.push(format!(
            "<< /Type /Outlines /First {} 0 R /Last {} 0 R /Count {} >>",
            first_item + top[0],
            first_item + top[top.len() - 1],
            outline.len()
        ));
        for (i, (title, _, page)) in outline.iter().enumerate() {
            let siblings = children(parents[i]);
            let pos = siblings.iter().position(|&s| s == i).unwrap();
            let mut entry = format!(
                "<< /Title ({}) /Parent {} 0 R /Dest [{} 0 R /XYZ 0 792 0]",
                escape(title),
                parents[i].map_or(outline_root, |p| first_item + p),
                3 + page - 1
            );
            if pos > 0 {
                entry.push_str(&format!(" /Prev {} 0 R", first_item + siblings[pos - 1]));
            }
            if pos + 1 < siblings.len() {
                entry.push_str(&format!(" /Next {} 0 R", first_item + siblings[pos + 1]));
            }
            let kids = children(Some(i));
            if !kids.is_empty() {
                entry.push_str(&format!(
                    " /First {} 0 R /Last {} 0 R /Count {}",
                    first_item + kids[0],
                    first_item + kids[kids.len() - 1],
                    kids.len()
                ));
            }
            entry.push_str(" >>");
            objects.push(entry);
        }
    }
    assemble(&objects)
}

/// Per page a 20pt heading then 11pt body lines, Helvetica WinAnsi.
pub fn layout_pdf(pages: &[(&str, Vec<String>)], landscape: bool) -> Vec<u8> {
    let (width, height) = if landscape { (792.0, 612.0) } else { (595.0, 842.0) };
    let specs: Vec<PageSpec> = pages
        .iter()
        .map(|(heading, lines)| {
            let mut ops = vec![(72.0, height - 80.0, 20.0, heading.to_string())];
            let mut y = height - 120.0;
            for line in lines {
                ops.push((72.0, y, 11.0, line.clone()));
                y -= 16.0;
            }
            PageSpec::Lines(ops)
        })
        .collect();
    build_pdf_with_outline(
        &specs,
        "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>",
        &[],
        (width, height),
    )
}

pub const DECK_TITLES: [&str; 8] = [
    "Revenue Overview",
    "Operating Expenses",
    "Customer Growth",
    "Product Roadmap",
    "Regional Performance",
    "Engineering Metrics",
    "Risk Factors",
    "Outlook and Guidance",
];

pub fn deck_pdf() -> Vec<u8> {
    let pages: Vec<(&str, Vec<String>)> = DECK_TITLES
        .iter()
        .enumerate()
        .map(|(i, title)| {
            (
                *title,
                (1..=5).map(|n| format!("Detail {n} for slide {} with a few more words of body text", i + 1)).collect(),
            )
        })
        .collect();
    layout_pdf(&pages, true)
}

pub fn memo_pdf() -> Vec<u8> {
    layout_pdf(
        &[
            ("Mission", vec!["The launch is named Skylark.".into()]),
            ("Budget", vec!["The budget is 420 euros.".into()]),
            ("Team", vec!["The lead is Ada.".into()]),
        ],
        false,
    )
}

/// A numbered report: a title page, a contents page, and sections with
/// subsections over body paragraphs, page numbers in the footer.
pub fn report_pdf() -> Vec<u8> {
    let body = |seed: usize| -> Vec<(f32, f32, f32, String)> {
        (0..6)
            .map(|row| {
                (
                    72.0,
                    600.0 - row as f32 * 14.0,
                    10.0,
                    format!("Body text line {row} of paragraph {seed} describes the findings in detail."),
                )
            })
            .collect()
    };
    let mut pages = Vec::new();
    let mut title_page = vec![(72.0, 700.0, 24.0, "Annual Systems Report".to_string())];
    title_page.extend((0..4).map(|r| {
        (
            72.0,
            640.0 - r as f32 * 14.0,
            10.0,
            format!("Opening paragraph line {r} explains why the report exists at all."),
        )
    }));
    pages.push(title_page);
    let sections = [
        ("1 Introduction", vec!["1.1 Scope", "1.2 Method"]),
        ("2 Results", vec!["2.1 Throughput", "2.2 Latency"]),
        ("3 Discussion", vec![]),
    ];
    let mut n = 2;
    for (section, subs) in sections {
        let mut page = vec![(72.0, 720.0, 16.0, section.to_string())];
        page.extend(body(n));
        pages.push(page);
        n += 1;
        for sub in subs {
            let mut page = vec![(72.0, 720.0, 13.0, sub.to_string())];
            page.extend(body(n));
            pages.push(page);
            n += 1;
        }
    }
    let specs: Vec<PageSpec> = pages
        .into_iter()
        .enumerate()
        .map(|(i, mut lines)| {
            lines.push((300.0, 40.0, 9.0, format!("{}", i + 1)));
            PageSpec::Lines(lines)
        })
        .collect();
    build_pdf(&specs, HELVETICA)
}
