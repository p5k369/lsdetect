//! Python bindings for the line segment detector.

use std::borrow::Cow;

use numpy::ndarray::{Array2, Array3};
use numpy::{
    IntoPyArray, PyArray2, PyArray3, PyReadonlyArray2, PyReadonlyArray3, PyUntypedArrayMethods,
};
use pyo3::prelude::*;

use crate::{detector, sample, warp};
use warp::Filter;

/// One detected line segment, in input-image coordinates.
#[pyclass(frozen, module = "lsdetect")]
struct Segment {
    #[pyo3(get)]
    x1: f64,
    #[pyo3(get)]
    y1: f64,
    #[pyo3(get)]
    x2: f64,
    #[pyo3(get)]
    y2: f64,
    #[pyo3(get)]
    width: f64,
    #[pyo3(get)]
    precision: f64,
    #[pyo3(get)]
    log_nfa: f64,
}

#[pymethods]
impl Segment {
    /// Direction in degrees, -90 to 90, 0 = horizontal.
    #[getter]
    fn angle(&self) -> f64 {
        self.as_inner().angle()
    }

    /// Length in pixels.
    #[getter]
    fn length(&self) -> f64 {
        self.as_inner().length()
    }

    fn __repr__(&self) -> String {
        format!(
            "Segment(({:.1}, {:.1})-({:.1}, {:.1}), width={:.1})",
            self.x1, self.y1, self.x2, self.y2, self.width
        )
    }
}

impl Segment {
    fn as_inner(&self) -> detector::Segment {
        detector::Segment {
            x1: self.x1,
            y1: self.y1,
            x2: self.x2,
            y2: self.y2,
            width: self.width,
            precision: self.precision,
            log_nfa: self.log_nfa,
        }
    }
}

/// The detector for one concrete gray depth, GIL released while it runs.
fn detect_typed<'py, T>(py: Python<'py>, gray: PyReadonlyArray2<'py, T>, scale: f64) -> Vec<Segment>
where
    T: sample::Gray + numpy::Element,
{
    let view = gray.as_array();
    let height = view.nrows();
    let width = view.ncols();
    let found = if let Some(direct) = view.as_slice() {
        py.detach(|| {
            detector::detect(direct, width, height, scale)
                .expect("the buffer length comes from the array's own shape")
        })
    } else {
        let data: Vec<T> = view.iter().copied().collect();
        py.detach(move || {
            detector::detect(&data, width, height, scale)
                .expect("the buffer length comes from the array's own shape")
        })
    };
    found
        .into_iter()
        .map(|s| Segment {
            x1: s.x1,
            y1: s.y1,
            x2: s.x2,
            y2: s.y2,
            width: s.width,
            precision: s.precision,
            log_nfa: s.log_nfa,
        })
        .collect()
}

/// Find the validated line segments of a grayscale image.
#[pyfunction]
#[pyo3(signature = (gray, scale = detector::SCALE))]
fn detect<'py>(py: Python<'py>, gray: &Bound<'py, PyAny>, scale: f64) -> PyResult<Vec<Segment>> {
    if let Ok(floats) = gray.extract::<PyReadonlyArray2<f64>>() {
        return Ok(detect_typed(py, floats, scale));
    }
    if let Ok(floats) = gray.extract::<PyReadonlyArray2<f32>>() {
        return Ok(detect_typed(py, floats, scale));
    }
    if let Ok(bytes) = gray.extract::<PyReadonlyArray2<u8>>() {
        return Ok(detect_typed(py, bytes, scale));
    }
    if let Ok(words) = gray.extract::<PyReadonlyArray2<u16>>() {
        return Ok(detect_typed(py, words, scale));
    }
    Err(pyo3::exceptions::PyTypeError::new_err(
        "detect expects a (height, width) array of uint8, uint16, \
         float32 or float64",
    ))
}

/// The warp for one concrete sample depth, GIL released while it runs.
#[allow(clippy::too_many_arguments)]
fn warp_typed<'py, T>(
    py: Python<'py>,
    src: PyReadonlyArray3<'py, T>,
    inverse: [f64; 9],
    scale: f64,
    off_x: f64,
    off_y: f64,
    filter: Filter,
    out_size: Option<(usize, usize)>,
) -> Bound<'py, PyArray3<T>>
where
    T: sample::Sample + numpy::Element,
{
    let view = src.as_array();
    let height = view.shape()[0];
    let width = view.shape()[1];
    let (out_width, out_height) = out_size.unwrap_or((width, height));
    let out = if let Some(direct) = view.as_slice() {
        py.detach(|| {
            warp::warp_rgb(
                direct, width, height, out_width, out_height, &inverse, scale, off_x, off_y, filter,
            )
            .expect("the buffer length comes from the array's own shape")
        })
    } else {
        let data: Vec<T> = view.iter().copied().collect();
        py.detach(move || {
            warp::warp_rgb(
                &data, width, height, out_width, out_height, &inverse, scale, off_x, off_y, filter,
            )
            .expect("the buffer length comes from the array's own shape")
        })
    };
    let array = Array3::from_shape_vec((out_height, out_width, 3), out).unwrap();
    array.into_pyarray(py)
}

/// The filter a Python caller named.
fn parse_filter(filter: &str) -> PyResult<Filter> {
    match filter {
        "bilinear" => Ok(Filter::Bilinear),
        "lanczos3" => Ok(Filter::Lanczos3),
        other => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "unknown filter {other:?}, expected \"bilinear\" or \"lanczos3\""
        ))),
    }
}

/// Perspective-warp an RGB image through an inverse homography.
#[pyfunction]
#[pyo3(signature = (src, inverse, scale, off_x, off_y, filter = "bilinear", out_size = None))]
#[allow(clippy::too_many_arguments)]
fn warp_rgb<'py>(
    py: Python<'py>,
    src: &Bound<'py, PyAny>,
    inverse: [f64; 9],
    scale: f64,
    off_x: f64,
    off_y: f64,
    filter: &str,
    out_size: Option<(usize, usize)>,
) -> PyResult<Bound<'py, PyAny>> {
    let kernel = parse_filter(filter)?;
    if let Ok(bytes) = src.extract::<PyReadonlyArray3<u8>>()
        && bytes.shape()[2] == 3
    {
        return Ok(
            warp_typed(py, bytes, inverse, scale, off_x, off_y, kernel, out_size).into_any(),
        );
    }
    if let Ok(words) = src.extract::<PyReadonlyArray3<u16>>()
        && words.shape()[2] == 3
    {
        return Ok(
            warp_typed(py, words, inverse, scale, off_x, off_y, kernel, out_size).into_any(),
        );
    }
    Err(pyo3::exceptions::PyTypeError::new_err(
        "warp_rgb expects a (height, width, 3) array of uint8 or uint16",
    ))
}

/// The remap for one concrete sample depth.
fn remap_typed<'py, T>(
    py: Python<'py>,
    src: PyReadonlyArray3<'py, T>,
    map_x: PyReadonlyArray2<'py, f32>,
    map_y: PyReadonlyArray2<'py, f32>,
    filter: Filter,
) -> PyResult<Bound<'py, PyArray3<T>>>
where
    T: sample::Sample + numpy::Element,
{
    let view = src.as_array();
    let height = view.shape()[0];
    let width = view.shape()[1];
    let xs = map_x.as_array();
    let ys = map_y.as_array();
    if xs.shape() != ys.shape() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "map_x and map_y must have the same shape",
        ));
    }
    let out_height = xs.shape()[0];
    let out_width = xs.shape()[1];
    let data = match view.as_slice() {
        Some(direct) => Cow::Borrowed(direct),
        None => Cow::Owned(view.iter().copied().collect()),
    };
    let xs = match xs.as_slice() {
        Some(direct) => Cow::Borrowed(direct),
        None => Cow::Owned(xs.iter().copied().collect()),
    };
    let ys = match ys.as_slice() {
        Some(direct) => Cow::Borrowed(direct),
        None => Cow::Owned(ys.iter().copied().collect()),
    };
    let out = py.detach(|| {
        warp::remap_rgb(
            &data, width, height, &xs, &ys, out_width, out_height, filter,
        )
        .expect("the buffer lengths come from the arrays' own shapes")
    });
    let array = Array3::from_shape_vec((out_height, out_width, 3), out).unwrap();
    Ok(array.into_pyarray(py))
}

/// Resample an RGB image through a coordinate map: output pixel (x, y)
/// takes its value from source position (map_x[y, x], map_y[y, x]).
#[pyfunction]
#[pyo3(signature = (src, map_x, map_y, filter = "bilinear"))]
fn remap_rgb<'py>(
    py: Python<'py>,
    src: &Bound<'py, PyAny>,
    map_x: PyReadonlyArray2<'py, f32>,
    map_y: PyReadonlyArray2<'py, f32>,
    filter: &str,
) -> PyResult<Bound<'py, PyAny>> {
    let kernel = parse_filter(filter)?;
    if let Ok(bytes) = src.extract::<PyReadonlyArray3<u8>>()
        && bytes.shape()[2] == 3
    {
        return Ok(remap_typed(py, bytes, map_x, map_y, kernel)?.into_any());
    }
    if let Ok(words) = src.extract::<PyReadonlyArray3<u16>>()
        && words.shape()[2] == 3
    {
        return Ok(remap_typed(py, words, map_x, map_y, kernel)?.into_any());
    }
    Err(pyo3::exceptions::PyTypeError::new_err(
        "remap_rgb expects a (height, width, 3) array of uint8 or uint16 \
         and float32 maps",
    ))
}

/// The gray remap for one concrete sample depth.
fn remap_gray_typed<'py, T>(
    py: Python<'py>,
    src: PyReadonlyArray2<'py, T>,
    map_x: PyReadonlyArray2<'py, f32>,
    map_y: PyReadonlyArray2<'py, f32>,
    filter: Filter,
) -> PyResult<Bound<'py, PyArray2<T>>>
where
    T: sample::Sample + numpy::Element,
{
    let view = src.as_array();
    let height = view.shape()[0];
    let width = view.shape()[1];
    let xs = map_x.as_array();
    let ys = map_y.as_array();
    if xs.shape() != ys.shape() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "map_x and map_y must have the same shape",
        ));
    }
    let out_height = xs.shape()[0];
    let out_width = xs.shape()[1];
    let data = match view.as_slice() {
        Some(direct) => Cow::Borrowed(direct),
        None => Cow::Owned(view.iter().copied().collect()),
    };
    let xs = match xs.as_slice() {
        Some(direct) => Cow::Borrowed(direct),
        None => Cow::Owned(xs.iter().copied().collect()),
    };
    let ys = match ys.as_slice() {
        Some(direct) => Cow::Borrowed(direct),
        None => Cow::Owned(ys.iter().copied().collect()),
    };
    let out = py.detach(|| {
        warp::remap_gray(
            &data, width, height, &xs, &ys, out_width, out_height, filter,
        )
        .expect("the buffer lengths come from the arrays' own shapes")
    });
    let array = Array2::from_shape_vec((out_height, out_width), out).unwrap();
    Ok(array.into_pyarray(py))
}

/// Resample a single-channel image (a mask) through a coordinate map.
#[pyfunction]
#[pyo3(signature = (src, map_x, map_y, filter = "bilinear"))]
fn remap_gray<'py>(
    py: Python<'py>,
    src: &Bound<'py, PyAny>,
    map_x: PyReadonlyArray2<'py, f32>,
    map_y: PyReadonlyArray2<'py, f32>,
    filter: &str,
) -> PyResult<Bound<'py, PyAny>> {
    let kernel = parse_filter(filter)?;
    if let Ok(bytes) = src.extract::<PyReadonlyArray2<u8>>() {
        return Ok(remap_gray_typed(py, bytes, map_x, map_y, kernel)?.into_any());
    }
    if let Ok(words) = src.extract::<PyReadonlyArray2<u16>>() {
        return Ok(remap_gray_typed(py, words, map_x, map_y, kernel)?.into_any());
    }
    Err(pyo3::exceptions::PyTypeError::new_err(
        "remap_gray expects a (height, width) array of uint8 or uint16 \
         and float32 maps",
    ))
}

#[pymodule]
fn lsdetect(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Segment>()?;
    m.add_function(wrap_pyfunction!(detect, m)?)?;
    m.add_function(wrap_pyfunction!(warp_rgb, m)?)?;
    m.add_function(wrap_pyfunction!(remap_rgb, m)?)?;
    m.add_function(wrap_pyfunction!(remap_gray, m)?)?;
    Ok(())
}
