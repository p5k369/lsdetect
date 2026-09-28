//! Perspective warp: a parallel bilinear homography resample.
//!
//! Generic image geometry, independent of the detector.

use rayon::prelude::*;

use crate::sample::{Sample, SizeMismatch};

/// For output pixel (x, y) the source position is
/// inverse * (x / scale + off_x, y / scale + off_y, 1), sampled
/// bilinearly. Pixels whose source falls outside the image stay
/// black. inverse is the 9 row-major entries of the 3x3 matrix.
pub fn warp_rgb<T: Sample>(
    src: &[T],
    width: usize,
    height: usize,
    inverse: &[f64; 9],
    scale: f64,
    off_x: f64,
    off_y: f64,
) -> Result<Vec<T>, SizeMismatch> {
    if src.len() != width * height * 3 {
        return Err(SizeMismatch {
            expected: width * height * 3,
            got: src.len(),
        });
    }
    let mut out = vec![T::from_f64(0.0); width * height * 3];
    let wf = width as f64 - 1.0;
    let hf = height as f64 - 1.0;
    out.par_chunks_mut(width * 3)
        .enumerate()
        .for_each(|(y, row)| {
            let py = y as f64 / scale + off_y;
            let base_x = inverse[1] * py + inverse[2];
            let base_y = inverse[4] * py + inverse[5];
            let base_w = inverse[7] * py + inverse[8];
            for x in 0..width {
                let px = x as f64 / scale + off_x;
                let w = inverse[6] * px + base_w;
                if w == 0.0 {
                    continue;
                }
                let sx = (inverse[0] * px + base_x) / w;
                let sy = (inverse[3] * px + base_y) / w;
                if sx < 0.0 || sx > wf || sy < 0.0 || sy > hf {
                    continue;
                }
                let x0 = sx.floor() as usize;
                let y0 = sy.floor() as usize;
                let x1 = (x0 + 1).min(width - 1);
                let y1 = (y0 + 1).min(height - 1);
                let fx = sx - x0 as f64;
                let fy = sy - y0 as f64;
                let w00 = (1.0 - fx) * (1.0 - fy);
                let w10 = fx * (1.0 - fy);
                let w01 = (1.0 - fx) * fy;
                let w11 = fx * fy;
                let i00 = (y0 * width + x0) * 3;
                let i10 = (y0 * width + x1) * 3;
                let i01 = (y1 * width + x0) * 3;
                let i11 = (y1 * width + x1) * 3;
                let o = x * 3;
                for c in 0..3 {
                    let value = src[i00 + c].to_f64() * w00
                        + src[i10 + c].to_f64() * w10
                        + src[i01 + c].to_f64() * w01
                        + src[i11 + c].to_f64() * w11;
                    row[o + c] = T::from_f64(value);
                }
            }
        });
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_wrong_buffer_length_is_refused() {
        let src = vec![0u8; 11];
        let result = warp_rgb(&src, 2, 2, &[0.0; 9], 1.0, 0.0, 0.0);
        assert_eq!(
            result.unwrap_err(),
            SizeMismatch {
                expected: 12,
                got: 11
            }
        );
    }
}
