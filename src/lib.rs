//! A line segment detector (LSD) in Rust, with optional Python bindings.

mod detector;
mod sample;
mod warp;

pub use detector::{SCALE, Segment, detect};
pub use sample::{Gray, Sample, SizeMismatch};
pub use warp::warp_rgb;

#[cfg(feature = "extension-module")]
mod python;
