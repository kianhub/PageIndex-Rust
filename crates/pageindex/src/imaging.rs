//! Image helpers for citation highlighting.

use image::{DynamicImage, ImageFormat, Rgba, RgbaImage, imageops};

use crate::error::{PageIndexError, Result};

/// Draw a translucent highlight over a bounding-box region.
///
/// `bbox` is `[x0, y0, x1, y1]` in units of `scale` from the top-left
/// corner, as `get_block()` returns it (default scale 1000).
/// Returns a new RGB PNG.
pub fn highlight_region(image: &[u8], bbox: &[f64], scale: f64) -> Result<Vec<u8>> {
    let decoded =
        image::load_from_memory(image).map_err(|e| PageIndexError::invalid(format!("Could not decode image: {e}")))?;
    highlight_decoded(&decoded, bbox, scale)
}

/// [`highlight_region`] over an already-decoded image.
pub fn highlight_decoded(image: &DynamicImage, bbox: &[f64], scale: f64) -> Result<Vec<u8>> {
    if bbox.len() != 4 {
        return Err(PageIndexError::invalid(format!("Invalid bbox: {bbox:?}")));
    }
    let [x0, y0, x1, y1] = [bbox[0], bbox[1], bbox[2], bbox[3]];
    if scale <= 0.0 || !(0.0 <= x0 && x0 < x1 && x1 <= scale && 0.0 <= y0 && y0 < y1 && y1 <= scale) {
        return Err(PageIndexError::invalid(format!("Invalid bbox for scale {scale}: {bbox:?}")));
    }
    let page = image.to_rgba8();
    let (width, height) = page.dimensions();
    let rect = (
        (x0 / scale * f64::from(width)) as i32,
        (y0 / scale * f64::from(height)) as i32,
        (x1 / scale * f64::from(width)) as i32,
        (y1 / scale * f64::from(height)) as i32,
    );
    let stroke = (width as f32 / 400.0).round().max(2.0) as i32;
    let mut overlay = RgbaImage::new(width, height);
    fill_rect(&mut overlay, rect, Rgba([255, 210, 0, 65]));
    stroke_rect(&mut overlay, rect, Rgba([255, 140, 0, 255]), stroke);
    let mut composed = page;
    imageops::overlay(&mut composed, &overlay, 0, 0);
    let rgb = DynamicImage::ImageRgba8(composed).to_rgb8();
    let mut out = Vec::new();
    DynamicImage::ImageRgb8(rgb)
        .write_to(&mut std::io::Cursor::new(&mut out), ImageFormat::Png)
        .map_err(|e| PageIndexError::invalid(format!("Could not encode highlight: {e}")))?;
    Ok(out)
}

fn fill_rect(img: &mut RgbaImage, (x0, y0, x1, y1): (i32, i32, i32, i32), color: Rgba<u8>) {
    let (w, h) = img.dimensions();
    let x0 = x0.clamp(0, w as i32);
    let y0 = y0.clamp(0, h as i32);
    let x1 = x1.clamp(0, w as i32);
    let y1 = y1.clamp(0, h as i32);
    for y in y0..y1 {
        for x in x0..x1 {
            img.put_pixel(x as u32, y as u32, color);
        }
    }
}

fn stroke_rect(img: &mut RgbaImage, (x0, y0, x1, y1): (i32, i32, i32, i32), color: Rgba<u8>, width: i32) {
    for i in 0..width {
        fill_rect(img, (x0 + i, y0 + i, x1 - i, y0 + i + 1), color);
        fill_rect(img, (x0 + i, y1 - i - 1, x1 - i, y1 - i), color);
        fill_rect(img, (x0 + i, y0 + i, x0 + i + 1, y1 - i), color);
        fill_rect(img, (x1 - i - 1, y0 + i, x1 - i, y1 - i), color);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{Rgb, RgbImage};

    #[test]
    fn highlights_a_region() {
        let mut img = RgbImage::new(100, 80);
        for p in img.pixels_mut() {
            *p = Rgb([10, 20, 30]);
        }
        let mut png = Vec::new();
        DynamicImage::ImageRgb8(img).write_to(&mut std::io::Cursor::new(&mut png), ImageFormat::Png).unwrap();
        let out = highlight_region(&png, &[100.0, 100.0, 400.0, 400.0], 1000.0).unwrap();
        let decoded = image::load_from_memory(&out).unwrap();
        assert_eq!(decoded.width(), 100);
        assert_eq!(decoded.height(), 80);
        assert!(highlight_region(&png, &[10.0, 10.0, 5.0, 20.0], 1000.0).is_err());
        assert!(highlight_region(&png, &[0.0, 0.0, 10.0, 10.0], 0.0).is_err());
    }
}
