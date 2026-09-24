//! A line segment detector (LSD) in Rust, with optional Python bindings.

mod detector;
mod warp;

pub use detector::{SCALE, Segment, detect};
pub use warp::warp_rgb;

#[cfg(feature = "extension-module")]
mod python;
