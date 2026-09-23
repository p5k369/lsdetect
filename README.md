# lsdetect

Line segment detector (LSD) for Rust and Python.

Implements ["LSD: a Line Segment Detector"](https://www.ipol.im/pub/art/2012/gjmr-lsd/) (Grompone von
Gioi, Jakubowicz, Morel, Randall. Image Processing On Line, 2012).

## Python

```python
import lsdetect

segments = lsdetect.detect(gray)

for s in segments:
    print(s.x1, s.y1, s.x2, s.y2, s.width, s.angle, s.length)
```

Build with [maturin](https://www.maturin.rs): `maturin develop
--release`. Python 3.11 or newer.

## Rust

```rust
let segments = lsdetect::detect(&gray, width, height, 0.8);
```

## License

MIT or Apache-2.0, at your option. Independent implementation from the article.
No code from the AGPL reference implementation was used.
