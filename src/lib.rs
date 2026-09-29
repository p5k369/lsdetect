//! A line segment detector (LSD) in Rust, with optional Python bindings.
//!
//! Implements "LSD: a Line Segment Detector" (Grompone von Gioi et al., IPOL
//! 2012) as an independent, permissively licensed implementation,
//! plus a parallel bilinear perspective warp for applying the
//! geometry the segments reveal.
//!
//! ```
//! # fn main() -> Result<(), lsdetect::SizeMismatch> {
//! // A vertical step edge through a 96x96 frame.
//! let mut image = vec![50u8; 96 * 96];
//! image.chunks_mut(96).for_each(|row| row[48..].fill(200));
//!
//! let segments = lsdetect::detect(&image, 96, 96, lsdetect::SCALE)?;
//! let longest = segments.iter().max_by(|a, b| {
//!     a.length().total_cmp(&b.length())
//! });
//! assert!(longest.is_some_and(|s| s.angle().abs() > 88.0));
//! # Ok(())
//! # }
//! ```

mod detector;
mod sample;
mod warp;

pub use detector::{SCALE, Segment, detect};
pub use sample::{Gray, Sample, SizeMismatch};
pub use warp::{Filter, warp_rgb};

#[cfg(feature = "extension-module")]
mod python;
