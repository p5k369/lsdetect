//! Line segment detection after the LSD article.
//!
//! Written from R. Grompone von Gioi, J. Jakubowicz, J.-M. Morel,
//! G. Randall: "LSD: a Line Segment Detector", Image Processing On
//! Line, 2012.

use rayon::prelude::*;
use std::f64::consts::PI;

/// Default working scale, shrinks the image first.
pub const SCALE: f64 = 0.8;
/// Anti-alias blur sigma, per unit of scale.
const SIGMA_PER_SCALE: f64 = 0.6;
/// Gray levels are quantized, so norms below q / sin(tau) carry no
/// trustworthy angle.
const GRAY_QUANTIZATION: f64 = 2.0;
/// Tau, degrees.
const ANGLE_TOLERANCE_DEG: f64 = 22.5;
/// Aligned points per rectangle area, below this the region is no
/// line.
const DENSITY_THRESHOLD: f64 = 0.7;
/// Survive when -log10(NFA) exceeds this. 0.0 is epsilon = 1.
const LOG_EPSILON: f64 = 0.0;
/// Bins for the pseudo-ordering, exactness is not needed.
const NORM_BINS: usize = 1024;

/// One detected line segment, in input-image coordinates.
#[derive(Clone, Copy, Debug)]
pub struct Segment {
    /// First endpoint x, in input-image pixels.
    pub x1: f64,
    /// First endpoint y, in input-image pixels.
    pub y1: f64,
    /// Second endpoint x, in input-image pixels.
    pub x2: f64,
    /// Second endpoint y, in input-image pixels.
    pub y2: f64,
    /// Segment width in pixels.
    pub width: f64,
    /// The angle tolerance the detection used, as a fraction of pi.
    pub precision: f64,
    /// Natural log of the number of false alarms.
    pub log_nfa: f64,
}

impl Segment {
    /// Direction in degrees, -90 to 90, 0 = horizontal.
    pub fn angle(&self) -> f64 {
        let mut a = (self.y2 - self.y1).atan2(self.x2 - self.x1).to_degrees();
        if a > 90.0 {
            a -= 180.0;
        }
        if a <= -90.0 {
            a += 180.0;
        }
        a
    }

    /// Length in pixels.
    pub fn length(&self) -> f64 {
        (self.x2 - self.x1).hypot(self.y2 - self.y1)
    }
}

/// One pixel of the gradient field. Packed: the neighbour test in
/// the growing loop is the hot path, one cache spot beats three
/// arrays.
#[derive(Clone, Copy, Default)]
struct Px {
    norm: f32,
    ux: f32,
    uy: f32,
}

/// A line-support region, summarized as a rectangle.
#[derive(Clone, Copy)]
struct Rect {
    x1: f64,
    y1: f64,
    x2: f64,
    y2: f64,
    width: f64,
    dx: f64,
    dy: f64,
    prec: f64,
    p: f64,
}

/// Find the validated line segments of a grayscale image.
pub fn detect(image: &[f64], width: usize, height: usize, scale: f64) -> Vec<Segment> {
    assert_eq!(image.len(), width * height);
    let owned;
    let (img, w, h) = if scale < 1.0 {
        owned = downscale(image, width, height, scale);
        (owned.0.as_slice(), owned.1, owned.2)
    } else {
        (image, width, height)
    };
    if w < 3 || h < 3 {
        return Vec::new();
    }
    let prec = PI * ANGLE_TOLERANCE_DEG / 180.0;
    let p = ANGLE_TOLERANCE_DEG / 180.0;
    let rho = (GRAY_QUANTIZATION / (ANGLE_TOLERANCE_DEG.to_radians()).sin()) as f32;
    let field = level_line_field(img, w, h);
    let log_nt = 2.5 * ((w as f64).log10() + (h as f64).log10()) + 11f64.log10();
    let min_region_pixels = (-log_nt / p.log10()).ceil() as usize;

    let seeds = seeds_by_norm(&field, rho);
    let mut used: Vec<bool> = field.iter().map(|px| px.norm <= rho).collect();
    let mut region: Vec<u32> = Vec::new();
    let mut segments = Vec::new();
    let back = if scale < 1.0 { 1.0 / scale } else { 1.0 };

    for &seed in &seeds {
        if used[seed as usize] {
            continue;
        }
        let mut region_angle = grow_region(seed, prec, &field, &mut used, &mut region, w, h);
        if region.len() < min_region_pixels {
            continue;
        }
        let mut rect = region_to_rect(&region, &field, region_angle, prec, p, w);
        let ok = refine(
            &mut region,
            &mut region_angle,
            prec,
            &mut rect,
            &mut used,
            &field,
            min_region_pixels,
            w,
            h,
        );
        if !ok {
            continue;
        }
        let log_nfa = improve_rect(&mut rect, &field, rho, log_nt, w, h);
        if log_nfa <= LOG_EPSILON {
            continue;
        }
        // The 2x2 gradient sits half a pixel to the lower right of
        // its anchor.
        segments.push(Segment {
            x1: (rect.x1 + 0.5) * back,
            y1: (rect.y1 + 0.5) * back,
            x2: (rect.x2 + 0.5) * back,
            y2: (rect.y2 + 0.5) * back,
            width: rect.width * back,
            precision: rect.p,
            log_nfa,
        });
    }
    segments.sort_by(|a, b| b.length().total_cmp(&a.length()));
    segments
}

/// Blur, then subsample.
fn downscale(image: &[f64], width: usize, height: usize, scale: f64) -> (Vec<f64>, usize, usize) {
    let sigma = SIGMA_PER_SCALE / scale;
    let radius = (sigma * (2.0 * 1000f64.ln()).sqrt()).ceil() as usize;
    let mut kernel: Vec<f64> = (0..=2 * radius)
        .map(|i| {
            let x = i as f64 - radius as f64;
            (-0.5 * (x / sigma).powi(2)).exp()
        })
        .collect();
    let total: f64 = kernel.iter().sum();
    for k in &mut kernel {
        *k /= total;
    }
    let clamp = |v: isize, hi: usize| v.clamp(0, hi as isize - 1) as usize;
    let mut horiz = vec![0.0; width * height];
    horiz
        .par_chunks_mut(width)
        .enumerate()
        .for_each(|(y, out)| {
            let row = &image[y * width..(y + 1) * width];
            for (i, &k) in kernel.iter().enumerate() {
                let off = i as isize - radius as isize;
                let (x0, x1) = if off < 0 {
                    ((-off) as usize, width)
                } else {
                    (0, width - (off as usize).min(width))
                };
                for x in x0..x1 {
                    out[x] += row[(x as isize + off) as usize] * k;
                }
                let edge = if off < 0 { row[0] } else { row[width - 1] } * k;
                for slot in out[..x0].iter_mut() {
                    *slot += edge;
                }
                for slot in out[x1..].iter_mut() {
                    *slot += edge;
                }
            }
        });
    let mut blurred = vec![0.0; width * height];
    blurred
        .par_chunks_mut(width)
        .enumerate()
        .for_each(|(y, out)| {
            for (i, &k) in kernel.iter().enumerate() {
                let sy = clamp(y as isize + i as isize - radius as isize, height);
                let src = &horiz[sy * width..(sy + 1) * width];
                for (slot, value) in out.iter_mut().zip(src) {
                    *slot += value * k;
                }
            }
        });
    let new_w = ((width as f64 * scale).floor() as usize).max(2);
    let new_h = ((height as f64 * scale).floor() as usize).max(2);
    let mut small = vec![0.0; new_w * new_h];
    for y in 0..new_h {
        let sy = ((y as f64 / scale) as usize).min(height - 1);
        for x in 0..new_w {
            let sx = ((x as f64 / scale) as usize).min(width - 1);
            small[y * new_w + x] = blurred[sy * width + sx];
        }
    }
    (small, new_w, new_h)
}

/// Gradient norm plus level-line direction.
fn level_line_field(image: &[f64], width: usize, height: usize) -> Vec<Px> {
    let mut field = vec![Px::default(); width * height];
    field
        .par_chunks_mut(width)
        .take(height - 1)
        .enumerate()
        .for_each(|(y, out)| {
            let row = &image[y * width..(y + 1) * width];
            let below = &image[(y + 1) * width..(y + 2) * width];
            for x in 0..width - 1 {
                let a = row[x];
                let b = row[x + 1];
                let c = below[x];
                let d = below[x + 1];
                let gx = (b + d - a - c) / 2.0;
                let gy = (c + d - a - b) / 2.0;
                let norm = gx.hypot(gy);
                if norm > 0.0 {
                    out[x] = Px {
                        norm: norm as f32,
                        ux: (-gy / norm) as f32,
                        uy: (gx / norm) as f32,
                    };
                }
            }
        });
    field
}

/// atan2 on demand. Almost nothing needs the actual angle.
fn angle_at(field: &[Px], i: usize) -> f64 {
    (field[i].uy as f64).atan2(field[i].ux as f64)
}

/// Strong pixels, strongest bins first.
fn seeds_by_norm(field: &[Px], rho: f32) -> Vec<u32> {
    let max = field.iter().map(|p| p.norm).fold(0.0f32, f32::max);
    if max <= 0.0 {
        return Vec::new();
    }
    let mut bins: Vec<Vec<u32>> = vec![Vec::new(); NORM_BINS];
    for (i, p) in field.iter().enumerate() {
        if p.norm > rho {
            let b = ((p.norm / max) * (NORM_BINS as f32 - 1.0)) as usize;
            bins[b.min(NORM_BINS - 1)].push(i as u32);
        }
    }
    bins.into_iter().rev().flatten().collect()
}

/// Angle distance on the full circle.
fn angle_diff(a: f64, b: f64) -> f64 {
    signed_angle_diff(a, b).abs()
}

/// Signed angle difference in (-pi, pi].
fn signed_angle_diff(a: f64, b: f64) -> f64 {
    let mut d = a - b;
    while d <= -PI {
        d += 2.0 * PI;
    }
    while d > PI {
        d -= 2.0 * PI;
    }
    d
}

/// Region growing. The mean direction updates with every added
/// pixel, which is what lets a slowly bending edge stay one region.
#[allow(clippy::too_many_arguments)]
fn grow_region(
    seed: u32,
    prec: f64,
    field: &[Px],
    used: &mut [bool],
    region: &mut Vec<u32>,
    width: usize,
    height: usize,
) -> f64 {
    region.clear();
    region.push(seed);
    used[seed as usize] = true;
    let cos_prec = prec.cos();
    let mut sum_dx = field[seed as usize].ux as f64;
    let mut sum_dy = field[seed as usize].uy as f64;
    let mut bar = cos_prec;
    let offsets: [isize; 8] = [
        -(width as isize) - 1,
        -(width as isize),
        -(width as isize) + 1,
        -1,
        1,
        width as isize - 1,
        width as isize,
        width as isize + 1,
    ];
    let mut i = 0;
    while i < region.len() {
        let p = region[i] as usize;
        let px = p % width;
        let py = p / width;
        // Interior pixels skip the bounds checks.
        if px >= 1 && py >= 1 && px + 1 < width && py + 1 < height {
            for &off in &offsets {
                let n = (p as isize + off) as usize;
                if used[n] {
                    continue;
                }
                let px_n = field[n];
                if px_n.ux as f64 * sum_dx + px_n.uy as f64 * sum_dy < bar {
                    continue;
                }
                used[n] = true;
                region.push(n as u32);
                sum_dx += px_n.ux as f64;
                sum_dy += px_n.uy as f64;
                bar = cos_prec * sum_dx.hypot(sum_dy);
            }
        } else {
            for ny in py.saturating_sub(1)..(py + 2).min(height) {
                for nx in px.saturating_sub(1)..(px + 2).min(width) {
                    let n = ny * width + nx;
                    if used[n] {
                        continue;
                    }
                    let px_n = field[n];
                    if px_n.ux as f64 * sum_dx + px_n.uy as f64 * sum_dy < bar {
                        continue;
                    }
                    used[n] = true;
                    region.push(n as u32);
                    sum_dx += px_n.ux as f64;
                    sum_dy += px_n.uy as f64;
                    bar = cos_prec * sum_dx.hypot(sum_dy);
                }
            }
        }
        i += 1;
    }
    sum_dy.atan2(sum_dx)
}

/// Weighted center, principal axis.
fn region_to_rect(
    region: &[u32],
    field: &[Px],
    region_angle: f64,
    prec: f64,
    p: f64,
    width: usize,
) -> Rect {
    let mut cx = 0.0;
    let mut cy = 0.0;
    let mut total = 0.0;
    for &r in region {
        let w = field[r as usize].norm as f64;
        cx += (r as usize % width) as f64 * w;
        cy += (r as usize / width) as f64 * w;
        total += w;
    }
    cx /= total;
    cy /= total;
    let theta = region_theta(region, field, cx, cy, region_angle, prec, width);
    let (dx, dy) = (theta.cos(), theta.sin());
    let mut l_min = 0.0f64;
    let mut l_max = 0.0f64;
    let mut w_min = 0.0f64;
    let mut w_max = 0.0f64;
    for &r in region {
        let rx = (r as usize % width) as f64 - cx;
        let ry = (r as usize / width) as f64 - cy;
        let l = rx * dx + ry * dy;
        let w = -rx * dy + ry * dx;
        l_min = l_min.min(l);
        l_max = l_max.max(l);
        w_min = w_min.min(w);
        w_max = w_max.max(w);
    }
    Rect {
        x1: cx + l_min * dx,
        y1: cy + l_min * dy,
        x2: cx + l_max * dx,
        y2: cy + l_max * dy,
        width: (w_max - w_min).max(1.0),
        dx,
        dy,
        prec,
        p,
    }
}

/// Principal axis of the region's second moments.
fn region_theta(
    region: &[u32],
    field: &[Px],
    cx: f64,
    cy: f64,
    region_angle: f64,
    prec: f64,
    width: usize,
) -> f64 {
    let mut ixx = 0.0;
    let mut iyy = 0.0;
    let mut ixy = 0.0;
    for &r in region {
        let w = field[r as usize].norm as f64;
        let rx = (r as usize % width) as f64 - cx;
        let ry = (r as usize / width) as f64 - cy;
        ixx += w * ry * ry;
        iyy += w * rx * rx;
        ixy -= w * rx * ry;
    }
    let smallest = (ixx + iyy - ((ixx - iyy).powi(2) + 4.0 * ixy * ixy).sqrt()) / 2.0;
    let mut theta = if (ixx - smallest).abs() > (iyy - smallest).abs() {
        (smallest - ixx).atan2(ixy)
    } else {
        ixy.atan2(smallest - iyy)
    };
    // Orient the axis like the region's level lines.
    if angle_diff(theta, region_angle) > prec {
        theta += PI;
    }
    theta
}

/// Rescue a too-sparse region.
#[allow(clippy::too_many_arguments)]
fn refine(
    region: &mut Vec<u32>,
    region_angle: &mut f64,
    prec: f64,
    rect: &mut Rect,
    used: &mut [bool],
    field: &[Px],
    min_size: usize,
    width: usize,
    height: usize,
) -> bool {
    if rect_density(region, rect) >= DENSITY_THRESHOLD {
        return true;
    }
    let seed = region[0];
    let sx = (seed as usize % width) as f64;
    let sy = (seed as usize / width) as f64;
    let seed_angle = angle_at(field, seed as usize);
    let mut sum = 0.0;
    let mut squares = 0.0;
    let mut n = 0.0;
    for &r in region.iter() {
        let rx = (r as usize % width) as f64;
        let ry = (r as usize / width) as f64;
        if (rx - sx).hypot(ry - sy) < rect.width {
            let d = signed_angle_diff(angle_at(field, r as usize), seed_angle);
            sum += d;
            squares += d * d;
            n += 1.0;
        }
    }
    let mean = sum / n;
    let variance = squares / n - mean * mean;
    let tau = 2.0 * variance.max(0.0).sqrt();
    for &r in region.iter() {
        used[r as usize] = false;
    }
    *region_angle = grow_region(seed, tau, field, used, region, width, height);
    if region.len() < min_size {
        return false;
    }
    *rect = region_to_rect(region, field, *region_angle, prec, rect.p, width);
    if rect_density(region, rect) >= DENSITY_THRESHOLD {
        return true;
    }
    reduce_radius(
        region,
        region_angle,
        prec,
        rect,
        used,
        field,
        min_size,
        width,
    )
}

/// Shrink the region radius until the rectangle is dense again.
#[allow(clippy::too_many_arguments)]
fn reduce_radius(
    region: &mut Vec<u32>,
    region_angle: &mut f64,
    prec: f64,
    rect: &mut Rect,
    used: &mut [bool],
    field: &[Px],
    min_size: usize,
    width: usize,
) -> bool {
    let seed = region[0];
    let sx = (seed as usize % width) as f64;
    let sy = (seed as usize / width) as f64;
    let mut radius = (sx - rect.x1)
        .hypot(sy - rect.y1)
        .max((sx - rect.x2).hypot(sy - rect.y2));
    loop {
        if rect_density(region, rect) >= DENSITY_THRESHOLD {
            return true;
        }
        radius *= 0.75;
        let mut i = 0;
        while i < region.len() {
            let r = region[i];
            let rx = (r as usize % width) as f64;
            let ry = (r as usize / width) as f64;
            if (rx - sx).hypot(ry - sy) > radius {
                used[r as usize] = false;
                region.swap_remove(i);
            } else {
                i += 1;
            }
        }
        if region.len() < min_size {
            return false;
        }
        *rect = region_to_rect(region, field, *region_angle, prec, rect.p, width);
    }
}

/// Region pixels per rectangle area.
fn rect_density(region: &[u32], rect: &Rect) -> f64 {
    let len = (rect.x2 - rect.x1).hypot(rect.y2 - rect.y1);
    region.len() as f64 / (len * rect.width).max(1.0)
}

/// n and k for the NFA test.
fn rect_counts(rect: &Rect, field: &[Px], rho: f32, width: usize, height: usize) -> (usize, usize) {
    let cx = (rect.x1 + rect.x2) / 2.0;
    let cy = (rect.y1 + rect.y2) / 2.0;
    let half_l = (rect.x2 - rect.x1).hypot(rect.y2 - rect.y1) / 2.0;
    let half_w = rect.width / 2.0;
    let reach = half_l.hypot(half_w);
    let x_lo = ((cx - reach).floor().max(0.0)) as usize;
    let y_lo = ((cy - reach).floor().max(0.0)) as usize;
    let x_hi = ((cx + reach).ceil() as usize).min(width - 1);
    let y_hi = ((cy + reach).ceil() as usize).min(height - 1);
    let cos_prec = rect.prec.cos() as f32;
    let dxf = rect.dx as f32;
    let dyf = rect.dy as f32;
    let mut total = 0;
    let mut aligned = 0;
    for y in y_lo..=y_hi {
        for x in x_lo..=x_hi {
            let rx = x as f64 - cx;
            let ry = y as f64 - cy;
            let l = rx * rect.dx + ry * rect.dy;
            if l.abs() > half_l {
                continue;
            }
            let w = -rx * rect.dy + ry * rect.dx;
            if w.abs() > half_w {
                continue;
            }
            total += 1;
            let px = field[y * width + x];
            if px.norm > rho && px.ux * dxf + px.uy * dyf >= cos_prec {
                aligned += 1;
            }
        }
    }
    (total, aligned)
}

/// -log10(NFA) of one rectangle.
fn rect_log_nfa(
    rect: &Rect,
    field: &[Px],
    rho: f32,
    log_nt: f64,
    width: usize,
    height: usize,
) -> f64 {
    let (total, aligned) = rect_counts(rect, field, rho, width, height);
    log_nfa(total, aligned, rect.p, log_nt)
}

/// Last chances for a failing rectangle.
fn improve_rect(
    rect: &mut Rect,
    field: &[Px],
    rho: f32,
    log_nt: f64,
    width: usize,
    height: usize,
) -> f64 {
    let mut best = rect_log_nfa(rect, field, rho, log_nt, width, height);
    if best > LOG_EPSILON {
        return best;
    }
    let delta = 0.5;
    let try_variant = |candidate: Rect, best: &mut f64, rect: &mut Rect| {
        let value = rect_log_nfa(&candidate, field, rho, log_nt, width, height);
        if value > *best {
            *best = value;
            *rect = candidate;
        }
    };
    let mut finer = *rect;
    for _ in 0..5 {
        finer.p /= 2.0;
        finer.prec = finer.p * PI;
        try_variant(finer, &mut best, rect);
    }
    if best > LOG_EPSILON {
        return best;
    }
    let mut slimmer = *rect;
    for _ in 0..5 {
        if slimmer.width - delta < 0.5 {
            break;
        }
        slimmer.width -= delta;
        try_variant(slimmer, &mut best, rect);
    }
    if best > LOG_EPSILON {
        return best;
    }
    for side in [1.0, -1.0] {
        let mut trimmed = *rect;
        for _ in 0..5 {
            if trimmed.width - delta < 0.5 {
                break;
            }
            trimmed.x1 += side * -trimmed.dy * delta / 2.0;
            trimmed.y1 += side * trimmed.dx * delta / 2.0;
            trimmed.x2 += side * -trimmed.dy * delta / 2.0;
            trimmed.y2 += side * trimmed.dx * delta / 2.0;
            trimmed.width -= delta;
            try_variant(trimmed, &mut best, rect);
        }
        if best > LOG_EPSILON {
            return best;
        }
    }
    let mut finest = *rect;
    for _ in 0..5 {
        finest.p /= 2.0;
        finest.prec = finest.p * PI;
        try_variant(finest, &mut best, rect);
    }
    best
}

/// -log10(NFA) for k aligned points out of n.
fn log_nfa(n: usize, k: usize, p: f64, log_nt: f64) -> f64 {
    if n == 0 || k == 0 {
        return -log_nt;
    }
    if n == k {
        return -log_nt - n as f64 * p.log10();
    }
    let nf = n as f64;
    let kf = k as f64;
    let odds = p / (1.0 - p);
    let log1 = log_gamma(nf + 1.0) - log_gamma(kf + 1.0) - log_gamma(nf - kf + 1.0)
        + kf * p.ln()
        + (nf - kf) * (1.0 - p).ln();
    let mut term = log1.exp();
    if term == 0.0 {
        if kf > nf * p {
            return -log1 / std::f64::consts::LN_10 - log_nt;
        }
        return -log_nt;
    }
    let mut tail = term;
    for i in k + 1..=n {
        let bin = (nf - i as f64 + 1.0) / i as f64;
        let mult = bin * odds;
        term *= mult;
        tail += term;
        if bin < 1.0 {
            let bound = term * ((1.0 - mult.powi((n - i + 1) as i32)) / (1.0 - mult) - 1.0);
            if bound < 0.1 * (-tail.log10() - log_nt).abs() * tail {
                break;
            }
        }
    }
    -tail.log10() - log_nt
}

/// ln of the gamma function, via the two approximations the
/// article's appendix prints (Lanczos below 15, Windschitl above,
/// the crossover is also the article's).
fn log_gamma(x: f64) -> f64 {
    if x >= 15.0 {
        return 0.918_938_533_204_673 + (x - 0.5) * x.ln() - x
            + 0.5 * x * ((x * (1.0 / x).sinh() + 1.0 / (810.0 * x.powi(6))).ln());
    }
    const Q: [f64; 7] = [
        75_122.633_153,
        80_916.627_895_2,
        36_308.295_147_7,
        8_687.245_297_05,
        1_168.926_494_79,
        83.867_604_342_4,
        2.506_628_275_11,
    ];
    let mut a = (x + 0.5) * (x + 5.5).ln() - (x + 5.5);
    let mut b = 0.0;
    for (n, q) in Q.iter().enumerate() {
        a -= (x + n as f64).ln();
        b += q * x.powi(n as i32);
    }
    a + b.ln()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn edge_image(width: usize, height: usize, split: usize) -> Vec<f64> {
        let mut img = vec![50.0; width * height];
        for y in 0..height {
            for x in split..width {
                img[y * width + x] = 200.0;
            }
        }
        img
    }

    #[test]
    fn vertical_edge_is_found() {
        let img = edge_image(120, 120, 60);
        let segs = detect(&img, 120, 120, SCALE);
        assert!(!segs.is_empty());
        let best = &segs[0];
        assert!(best.angle().abs() > 88.0, "angle {}", best.angle());
        assert!(best.length() > 60.0, "length {}", best.length());
        assert!(best.x1 > 45.0 && best.x1 < 75.0, "x {}", best.x1);
    }

    #[test]
    fn flat_image_is_empty() {
        let img = vec![128.0; 100 * 100];
        assert!(detect(&img, 100, 100, SCALE).is_empty());
    }
}

#[cfg(test)]
mod bench {
    use super::*;
    use std::time::Instant;

    fn load(path: &str) -> (Vec<f64>, usize, usize) {
        let bytes = std::fs::read(path).unwrap();
        let w = u64::from_le_bytes(bytes[0..8].try_into().unwrap()) as usize;
        let h = u64::from_le_bytes(bytes[8..16].try_into().unwrap()) as usize;
        let img = bytes[16..]
            .chunks_exact(8)
            .map(|c| f64::from_le_bytes(c.try_into().unwrap()))
            .collect();
        (img, w, h)
    }

    #[test]
    #[ignore]
    fn stages() {
        let path = std::env::var("LSDETECT_BENCH").unwrap();
        let (img, w, h) = load(&path);
        let t = Instant::now();
        let (small, sw, sh) = downscale(&img, w, h, SCALE);
        eprintln!("downscale   {:7.1?}", t.elapsed());
        let t = Instant::now();
        let field = level_line_field(&small, sw, sh);
        eprintln!("field       {:7.1?}", t.elapsed());
        let rho = (GRAY_QUANTIZATION / (ANGLE_TOLERANCE_DEG.to_radians()).sin()) as f32;
        let t = Instant::now();
        let seeds = seeds_by_norm(&field, rho);
        eprintln!("seeds       {:7.1?} ({})", t.elapsed(), seeds.len());
        let prec = PI * ANGLE_TOLERANCE_DEG / 180.0;
        let p = ANGLE_TOLERANCE_DEG / 180.0;
        let log_nt = 2.5 * ((sw as f64).log10() + (sh as f64).log10()) + 11f64.log10();
        let min_region_pixels = (-log_nt / p.log10()).ceil() as usize;
        let mut used: Vec<bool> = field.iter().map(|px| px.norm <= rho).collect();
        let mut region = Vec::new();
        let (mut t_grow, mut t_rect, mut t_nfa) = (0.0f64, 0.0f64, 0.0f64);
        let mut kept = 0;
        for &seed in &seeds {
            if used[seed as usize] {
                continue;
            }
            let t = Instant::now();
            let mut region_angle = grow_region(seed, prec, &field, &mut used, &mut region, sw, sh);
            t_grow += t.elapsed().as_secs_f64();
            if region.len() < min_region_pixels {
                continue;
            }
            let t = Instant::now();
            let mut rect = region_to_rect(&region, &field, region_angle, prec, p, sw);
            let ok = refine(
                &mut region,
                &mut region_angle,
                prec,
                &mut rect,
                &mut used,
                &field,
                min_region_pixels,
                sw,
                sh,
            );
            t_rect += t.elapsed().as_secs_f64();
            if !ok {
                continue;
            }
            let t = Instant::now();
            let v = improve_rect(&mut rect, &field, rho, log_nt, sw, sh);
            t_nfa += t.elapsed().as_secs_f64();
            if v > LOG_EPSILON {
                kept += 1;
            }
        }
        eprintln!(
            "grow        {:6.1} ms\nrect+refine {:6.1} ms\nnfa+improve {:6.1} ms\nkept {}",
            t_grow * 1e3,
            t_rect * 1e3,
            t_nfa * 1e3,
            kept
        );
    }
}
