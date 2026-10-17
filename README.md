# autodiff-cpp

[![CI](https://github.com/laviserva/autodiff-cpp/actions/workflows/ci.yml/badge.svg)](https://github.com/laviserva/autodiff-cpp/actions/workflows/ci.yml)

A reverse-mode automatic differentiation engine written from scratch in modern C++ (C++20),
built step by step toward competitive performance.

## Status

Early stage. Currently implemented:

- N-dimensional `Tensor` with row-major strides and value semantics (copy / move).
- Reproducible build with CMake presets, tested on Linux (GCC) and Windows (MSVC) in CI.

Autodiff itself is not implemented yet — see [Roadmap](#roadmap).

## Build and test

Requirements: a C++20 compiler (GCC, Clang or MSVC) and CMake ≥ 3.25.

```bash
cmake --preset release
cmake --build --preset release
ctest --preset release
```

The same three commands work on Linux and Windows.

## Presets

| Preset           | Purpose                                                        |
|------------------|----------------------------------------------------------------|
| `debug`          | No optimization, full debug info.                              |
| `release`        | Full optimization (`-O3` / `/O2`), asserts disabled.           |
| `relwithdebinfo` | Optimized with debug symbols — used for profiling.             |
| `asan`           | AddressSanitizer + UBSan. **Linux only**: MSVC has no UBSan.   |

## Layout

```
include/     public headers (Tensor)
src/         executable entry point
tests/       test suite (run by ctest)
benchmarks/  performance benchmarks (planned)
cmake/       CMake helper modules
```

## Roadmap

- [x] Toolchain, presets, cross-platform CI
- [x] N-dimensional Tensor with strides and value semantics
- [ ] `const` correctness, ownership rules, `clang-tidy` in CI
- [ ] Error-handling policy (`optional` / `expected` / exceptions)
- [ ] Computational graph and reverse-mode backpropagation
- [ ] Benchmarks against reference implementations

## License

TBD.