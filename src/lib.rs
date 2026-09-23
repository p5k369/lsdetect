//! Python bindings for the lsdetect line segment detector.

use numpy::PyReadonlyArray2;
use pyo3::prelude::*;

mod detector;

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
            log_nfa: self.log_nfa,
        }
    }
}

/// Find the validated line segments of a grayscale image.
#[pyfunction]
#[pyo3(signature = (gray, scale = detector::SCALE))]
fn detect(py: Python<'_>, gray: PyReadonlyArray2<'_, f64>, scale: f64) -> Vec<Segment> {
    let view = gray.as_array();
    let height = view.nrows();
    let width = view.ncols();
    let data: Vec<f64> = view.iter().copied().collect();
    let found = py.detach(move || detector::detect(&data, width, height, scale));
    found
        .into_iter()
        .map(|s| Segment {
            x1: s.x1,
            y1: s.y1,
            x2: s.x2,
            y2: s.y2,
            width: s.width,
            log_nfa: s.log_nfa,
        })
        .collect()
}

#[pymodule]
fn lsdetect(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Segment>()?;
    m.add_function(wrap_pyfunction!(detect, m)?)?;
    Ok(())
}
