//! Perspective warp: a parallel homography resample.
//!
//! Generic image geometry, independent of the detector.

use rayon::prelude::*;

use crate::sample::{Sample, SizeMismatch};

/// How the warp samples between source pixels.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Filter {
    /// Four taps, fastest, softens the finest detail.
    Bilinear,
    /// Thirty-six taps, keeps fine detail, may ring at hard edges.
    Lanczos3,
}

/// Phase resolution of the precomputed Lanczos weight table.
const PHASES: usize = 1024;

fn lanczos3(t: f64) -> f64 {
    if t == 0.0 {
        return 1.0;
    }
    if t.abs() >= 3.0 {
        return 0.0;
    }
    let p = std::f64::consts::PI * t;
    (p.sin() / p) * ((p / 3.0).sin() / (p / 3.0))
}

/// The six tap weights for every quantized subpixel phase, normalized
/// so each row sums to one.
fn lanczos3_table() -> Vec<[f64; 6]> {
    (0..=PHASES)
        .map(|phase| {
            let fx = phase as f64 / PHASES as f64;
            let mut weights = [0.0; 6];
            let mut sum = 0.0;
            for (idx, weight) in weights.iter_mut().enumerate() {
                *weight = lanczos3(fx - (idx as f64 - 2.0));
                sum += *weight;
            }
            for weight in &mut weights {
                *weight /= sum;
            }
            weights
        })
        .collect()
}

/// One bilinear sample at (sx, sy) into out[0..C] of a C-channel image.
#[inline]
fn sample_bilinear<T: Sample, const C: usize>(
    src: &[T],
    width: usize,
    height: usize,
    sx: f64,
    sy: f64,
    out: &mut [T],
) {
    if sx < -0.5 || sx > width as f64 - 0.5 || sy < -0.5 || sy > height as f64 - 0.5 {
        return;
    }
    let fx0 = sx.floor();
    let fy0 = sy.floor();
    let fx = sx - fx0;
    let fy = sy - fy0;
    let x0 = (fx0 as isize).clamp(0, width as isize - 1) as usize;
    let y0 = (fy0 as isize).clamp(0, height as isize - 1) as usize;
    let x1 = (x0 + 1).min(width - 1);
    let y1 = (y0 + 1).min(height - 1);
    let w00 = (1.0 - fx) * (1.0 - fy);
    let w10 = fx * (1.0 - fy);
    let w01 = (1.0 - fx) * fy;
    let w11 = fx * fy;
    let i00 = (y0 * width + x0) * C;
    let i10 = (y0 * width + x1) * C;
    let i01 = (y1 * width + x0) * C;
    let i11 = (y1 * width + x1) * C;
    for c in 0..C {
        let value = src[i00 + c].to_f64() * w00
            + src[i10 + c].to_f64() * w10
            + src[i01 + c].to_f64() * w01
            + src[i11 + c].to_f64() * w11;
        out[c] = T::from_f64(value);
    }
}

/// One Lanczos3 sample at (sx, sy) into out[0..C] of a C-channel image.
#[inline]
fn sample_lanczos3<T: Sample, const C: usize>(
    src: &[T],
    width: usize,
    height: usize,
    table: &[[f64; 6]],
    sx: f64,
    sy: f64,
    out: &mut [T],
) {
    if sx < -0.5 || sx > width as f64 - 0.5 || sy < -0.5 || sy > height as f64 - 0.5 {
        return;
    }
    let last_x = (width - 1) as isize;
    let last_y = (height - 1) as isize;
    let x0 = sx.floor();
    let y0 = sy.floor();
    let wx = &table[((sx - x0) * PHASES as f64).round() as usize];
    let wy = &table[((sy - y0) * PHASES as f64).round() as usize];
    let x0 = x0 as isize;
    let y0 = y0 as isize;
    let mut acc = [0.0f64; C];
    if x0 >= 2 && x0 + 3 <= last_x && y0 >= 2 && y0 + 3 <= last_y {
        let x0 = x0 as usize;
        let y0 = y0 as usize;
        for (j, wyj) in wy.iter().enumerate() {
            let start = ((y0 + j - 2) * width + x0 - 2) * C;
            let taps = &src[start..start + 6 * C];
            for (i, wxi) in wx.iter().enumerate() {
                let weight = wyj * wxi;
                for c in 0..C {
                    acc[c] += taps[i * C + c].to_f64() * weight;
                }
            }
        }
    } else {
        for (j, wyj) in wy.iter().enumerate() {
            let yi = (y0 + j as isize - 2).clamp(0, last_y) as usize;
            let row_base = yi * width;
            for (i, wxi) in wx.iter().enumerate() {
                let xi = (x0 + i as isize - 2).clamp(0, last_x) as usize;
                let tap = (row_base + xi) * C;
                let weight = wyj * wxi;
                for c in 0..C {
                    acc[c] += src[tap + c].to_f64() * weight;
                }
            }
        }
    }
    for c in 0..C {
        out[c] = T::from_f64(acc[c]);
    }
}

/// Resample an RGB image through a coordinate map: output pixel
/// (x, y) takes its value from source position (map_x[y][x],
/// map_y[y][x]), in source pixel units, sampled with the chosen
/// filter. The maps are row-major out_width by out_height. Pixels
/// whose source falls outside the image stay black.
///
/// ```
/// use lsdetect::Filter;
///
/// let frame = vec![1000u16; 4 * 4 * 3];
/// let xs: Vec<f32> = (0..16).map(|i| (i % 4) as f32).collect();
/// let ys: Vec<f32> = (0..16).map(|i| (i / 4) as f32).collect();
/// let out = lsdetect::remap_rgb(&frame, 4, 4, &xs, &ys, 4, 4, Filter::Lanczos3);
/// assert_eq!(out.unwrap(), frame);
/// ```
///
/// # Errors
///
/// [`SizeMismatch`] when the image does not hold width * height * 3
/// samples or a map does not hold out_width * out_height entries.
#[allow(clippy::too_many_arguments)]
pub fn remap_rgb<T: Sample>(
    src: &[T],
    width: usize,
    height: usize,
    map_x: &[f32],
    map_y: &[f32],
    out_width: usize,
    out_height: usize,
    filter: Filter,
) -> Result<Vec<T>, SizeMismatch> {
    remap::<T, 3>(
        src, width, height, map_x, map_y, out_width, out_height, filter,
    )
}

/// [`remap_rgb`] for a single-channel image, such as a mask that has
/// to follow the geometry its picture went through.
///
/// ```
/// use lsdetect::Filter;
///
/// let mask = vec![255u8; 4 * 4];
/// let xs: Vec<f32> = (0..16).map(|i| (i % 4) as f32).collect();
/// let ys: Vec<f32> = (0..16).map(|i| (i / 4) as f32).collect();
/// let out = lsdetect::remap_gray(&mask, 4, 4, &xs, &ys, 4, 4, Filter::Bilinear);
/// assert_eq!(out.unwrap(), mask);
/// ```
///
/// # Errors
///
/// [`SizeMismatch`] when the image does not hold width * height
/// samples or a map does not hold out_width * out_height entries.
#[allow(clippy::too_many_arguments)]
pub fn remap_gray<T: Sample>(
    src: &[T],
    width: usize,
    height: usize,
    map_x: &[f32],
    map_y: &[f32],
    out_width: usize,
    out_height: usize,
    filter: Filter,
) -> Result<Vec<T>, SizeMismatch> {
    remap::<T, 1>(
        src, width, height, map_x, map_y, out_width, out_height, filter,
    )
}

#[allow(clippy::too_many_arguments)]
fn remap<T: Sample, const C: usize>(
    src: &[T],
    width: usize,
    height: usize,
    map_x: &[f32],
    map_y: &[f32],
    out_width: usize,
    out_height: usize,
    filter: Filter,
) -> Result<Vec<T>, SizeMismatch> {
    if src.len() != width * height * C {
        return Err(SizeMismatch {
            expected: width * height * C,
            got: src.len(),
        });
    }
    for map in [map_x, map_y] {
        if map.len() != out_width * out_height {
            return Err(SizeMismatch {
                expected: out_width * out_height,
                got: map.len(),
            });
        }
    }
    let table = match filter {
        Filter::Bilinear => Vec::new(),
        Filter::Lanczos3 => lanczos3_table(),
    };
    let mut out = vec![T::from_f64(0.0); out_width * out_height * C];
    out.par_chunks_mut(out_width * C)
        .enumerate()
        .for_each(|(y, row)| {
            let xs = &map_x[y * out_width..(y + 1) * out_width];
            let ys = &map_y[y * out_width..(y + 1) * out_width];
            for (x, (sx, sy)) in xs.iter().zip(ys).enumerate() {
                let pixel = &mut row[x * C..x * C + C];
                match filter {
                    Filter::Bilinear => {
                        sample_bilinear::<T, C>(src, width, height, *sx as f64, *sy as f64, pixel);
                    }
                    Filter::Lanczos3 => {
                        sample_lanczos3::<T, C>(
                            src, width, height, &table, *sx as f64, *sy as f64, pixel,
                        );
                    }
                }
            }
        });
    Ok(out)
}

/// For output pixel (x, y) the source position is
/// inverse * (x / scale + off_x, y / scale + off_y, 1), sampled with
/// the chosen filter into an out_width by out_height frame. Pixels
/// whose source falls outside the image stay black. inverse is the 9
/// row-major entries of the 3x3 matrix.
///
/// ```
/// use lsdetect::Filter;
///
/// let frame = vec![1000u16; 4 * 4 * 3];
/// let identity = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
/// let out = lsdetect::warp_rgb(
///     &frame, 4, 4, 4, 4, &identity, 1.0, 0.0, 0.0, Filter::Lanczos3,
/// );
/// assert_eq!(out.unwrap(), frame);
/// ```
///
/// # Errors
///
/// [`SizeMismatch`] when the buffer does not hold width * height * 3
/// samples.
#[allow(clippy::too_many_arguments)]
pub fn warp_rgb<T: Sample>(
    src: &[T],
    width: usize,
    height: usize,
    out_width: usize,
    out_height: usize,
    inverse: &[f64; 9],
    scale: f64,
    off_x: f64,
    off_y: f64,
    filter: Filter,
) -> Result<Vec<T>, SizeMismatch> {
    if src.len() != width * height * 3 {
        return Err(SizeMismatch {
            expected: width * height * 3,
            got: src.len(),
        });
    }
    match filter {
        Filter::Bilinear => Ok(warp_bilinear(
            src, width, height, out_width, out_height, inverse, scale, off_x, off_y,
        )),
        Filter::Lanczos3 => Ok(warp_lanczos3(
            src, width, height, out_width, out_height, inverse, scale, off_x, off_y,
        )),
    }
}

#[allow(clippy::too_many_arguments)]
fn warp_bilinear<T: Sample>(
    src: &[T],
    width: usize,
    height: usize,
    out_width: usize,
    out_height: usize,
    inverse: &[f64; 9],
    scale: f64,
    off_x: f64,
    off_y: f64,
) -> Vec<T> {
    let mut out = vec![T::from_f64(0.0); out_width * out_height * 3];
    out.par_chunks_mut(out_width * 3)
        .enumerate()
        .for_each(|(y, row)| {
            let py = y as f64 / scale + off_y;
            let base_x = inverse[1] * py + inverse[2];
            let base_y = inverse[4] * py + inverse[5];
            let base_w = inverse[7] * py + inverse[8];
            for x in 0..out_width {
                let px = x as f64 / scale + off_x;
                let w = inverse[6] * px + base_w;
                if w == 0.0 {
                    continue;
                }
                let sx = (inverse[0] * px + base_x) / w;
                let sy = (inverse[3] * px + base_y) / w;
                sample_bilinear::<T, 3>(src, width, height, sx, sy, &mut row[x * 3..x * 3 + 3]);
            }
        });
    out
}

#[allow(clippy::too_many_arguments)]
fn warp_lanczos3<T: Sample>(
    src: &[T],
    width: usize,
    height: usize,
    out_width: usize,
    out_height: usize,
    inverse: &[f64; 9],
    scale: f64,
    off_x: f64,
    off_y: f64,
) -> Vec<T> {
    let table = lanczos3_table();
    let mut out = vec![T::from_f64(0.0); out_width * out_height * 3];
    out.par_chunks_mut(out_width * 3)
        .enumerate()
        .for_each(|(y, row)| {
            let py = y as f64 / scale + off_y;
            let base_x = inverse[1] * py + inverse[2];
            let base_y = inverse[4] * py + inverse[5];
            let base_w = inverse[7] * py + inverse[8];
            for x in 0..out_width {
                let px = x as f64 / scale + off_x;
                let w = inverse[6] * px + base_w;
                if w == 0.0 {
                    continue;
                }
                let sx = (inverse[0] * px + base_x) / w;
                let sy = (inverse[3] * px + base_y) / w;
                sample_lanczos3::<T, 3>(
                    src,
                    width,
                    height,
                    &table,
                    sx,
                    sy,
                    &mut row[x * 3..x * 3 + 3],
                );
            }
        });
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_wrong_buffer_length_is_refused() {
        let src = vec![0u8; 11];
        let result = warp_rgb(&src, 2, 2, 2, 2, &[0.0; 9], 1.0, 0.0, 0.0, Filter::Bilinear);
        assert_eq!(
            result.unwrap_err(),
            SizeMismatch {
                expected: 12,
                got: 11
            }
        );
    }

    #[test]
    fn lanczos_is_exact_on_the_identity() {
        let identity = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let src: Vec<u16> = (0..4 * 4 * 3).map(|v| (v * 3001) as u16).collect();
        let out = warp_rgb(&src, 4, 4, 4, 4, &identity, 1.0, 0.0, 0.0, Filter::Lanczos3);
        assert_eq!(out.unwrap(), src);
    }
}

#[cfg(test)]
mod remap_tests {
    use super::*;

    #[test]
    fn an_identity_map_is_exact() {
        let src: Vec<u16> = (0..4 * 4 * 3).map(|v| (v * 3001) as u16).collect();
        let xs: Vec<f32> = (0..16).map(|i| (i % 4) as f32).collect();
        let ys: Vec<f32> = (0..16).map(|i| (i / 4) as f32).collect();
        for filter in [Filter::Bilinear, Filter::Lanczos3] {
            let out = remap_rgb(&src, 4, 4, &xs, &ys, 4, 4, filter);
            assert_eq!(out.unwrap(), src);
        }
    }

    #[test]
    fn a_map_that_leaves_the_image_stays_black() {
        let src = vec![200u8; 2 * 2 * 3];
        let out = remap_rgb(&src, 2, 2, &[9.0, 0.0], &[9.0, 0.0], 2, 1, Filter::Bilinear);
        assert_eq!(out.unwrap(), vec![0, 0, 0, 200, 200, 200]);
    }

    #[test]
    fn a_gray_identity_map_is_exact_and_sized_by_the_maps() {
        let src: Vec<u8> = (0..16).map(|v| (v * 17) as u8).collect();
        let xs: Vec<f32> = (0..16).map(|i| (i % 4) as f32).collect();
        let ys: Vec<f32> = (0..16).map(|i| (i / 4) as f32).collect();
        let out = remap_gray(&src, 4, 4, &xs, &ys, 4, 4, Filter::Lanczos3);
        assert_eq!(out.unwrap(), src);
        let half = remap_gray(&src, 4, 4, &xs[..8], &ys[..8], 4, 2, Filter::Bilinear);
        assert_eq!(half.unwrap(), src[..8]);
    }

    #[test]
    fn a_short_map_is_refused() {
        let src = vec![0u8; 2 * 2 * 3];
        let result = remap_rgb(&src, 2, 2, &[0.0; 3], &[0.0; 4], 2, 2, Filter::Bilinear);
        assert_eq!(
            result.unwrap_err(),
            SizeMismatch {
                expected: 4,
                got: 3
            }
        );
    }
}

#[cfg(test)]
mod out_size_tests {
    use super::*;

    #[test]
    fn the_output_frame_can_differ_from_the_source() {
        // A 2x2 window into a 4x4 source, offset by one pixel.
        let identity = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let src: Vec<u16> = (0..4 * 4 * 3).map(|v| v as u16).collect();
        let out = warp_rgb(&src, 4, 4, 2, 2, &identity, 1.0, 1.0, 1.0, Filter::Bilinear).unwrap();
        assert_eq!(out.len(), 2 * 2 * 3);
        assert_eq!(out[0], src[(4 + 1) * 3]);
    }
}
