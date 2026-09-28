//! The sample depths the crate's image inputs come in.

use std::fmt;

mod private {
    pub trait Sealed {}
    impl Sealed for u8 {}
    impl Sealed for u16 {}
    impl Sealed for f32 {}
    impl Sealed for f64 {}
}

/// A gray sample the detector can read.
pub trait Gray: Copy + Send + Sync + private::Sealed {
    /// The sample on the gray scale the detector's thresholds
    /// are calibrated for.
    fn to_gray(self) -> f64;

    /// The buffer itself when it already holds f64 grays.
    fn as_f64_slice(image: &[Self]) -> Option<&[f64]> {
        let _ = image;
        None
    }
}

impl Gray for u8 {
    fn to_gray(self) -> f64 {
        self as f64
    }
}

impl Gray for u16 {
    fn to_gray(self) -> f64 {
        self as f64 / 257.0
    }
}

impl Gray for f32 {
    fn to_gray(self) -> f64 {
        self as f64
    }
}

impl Gray for f64 {
    fn to_gray(self) -> f64 {
        self
    }

    fn as_f64_slice(image: &[Self]) -> Option<&[f64]> {
        Some(image)
    }
}

/// An RGB sample depth the warp can resample.
pub trait Sample: Copy + Send + Sync + private::Sealed {
    /// The sample as the float the interpolation runs on.
    fn to_f64(self) -> f64;
    /// The rounded interpolation result, back at this depth.
    fn from_f64(value: f64) -> Self;
}

impl Sample for u8 {
    fn to_f64(self) -> f64 {
        self as f64
    }

    fn from_f64(value: f64) -> Self {
        (value + 0.5) as u8
    }
}

impl Sample for u16 {
    fn to_f64(self) -> f64 {
        self as f64
    }

    fn from_f64(value: f64) -> Self {
        (value + 0.5) as u16
    }
}

/// The pixel buffer does not match the stated dimensions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SizeMismatch {
    /// Samples the dimensions demand.
    pub expected: usize,
    /// Samples the buffer holds.
    pub got: usize,
}

impl fmt::Display for SizeMismatch {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "buffer holds {} samples, the dimensions demand {}",
            self.got, self.expected
        )
    }
}

impl std::error::Error for SizeMismatch {}
