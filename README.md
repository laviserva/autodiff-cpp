# autodiff-cpp

[![CI](https://github.com/laviserva/autodiff-cpp/actions/workflows/ci.yml/badge.svg)](https://github.com/laviserva/autodiff-cpp/actions/workflows/ci.yml)

A reverse-mode automatic differentiation engine written from scratch in modern C++ (C++20),
built step by step toward competitive performance.

## Status

Early stage. Currently implemented:

- N-dimensional `Tensor` with row-major strides, `std::span` views, and value semantics (copy / move).
- Dual-path element access: unchecked `operator()` (zero-overhead hot path) and checked `at()` (exception-safe).
- Reproducible build with CMake presets and `clang-tidy`, tested on Linux (GCC) and Windows (MSVC) in CI.

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
docs/        architecture and ownership rules (docs/ownership.md)
benchmarks/  performance benchmarks (planned)
cmake/       CMake helper modules
```

## Error Handling Policy

Following the C++ Zero-Overhead Principle and the design of the STL and PyTorch (`libtorch`), error handling is stratified by error category and performance requirements:

- **`operator()` (Unchecked / Fast Path):** Designed for inner numerical loops. Performs zero runtime branching in `Release` builds. Preconditions (rank and bounds) are guarded via `assert()` in `Debug` builds to catch programmer bugs early without penalizing production speed.
- **`at()` (Checked / Safe Path):** Validates rank and coordinate bounds at runtime. Throws `std::invalid_argument` if the number of indices does not match `ndim()`, and `std::out_of_range` if any index exceeds `shape()`. Construction guards against integer overflow in element counts via `std::overflow_error`.
- **`std::optional` & `std::expected`:** Intentionally omitted from scalar element indexing (`operator()` / `at()`) to preserve reference semantics (`double&`) and avoid wrapper branching overhead on hot paths. `std::optional` is reserved for queries where absence is a normal state (e.g., searches), and `std::expected` is reserved for high-level tensor transformations where shape or dtype mismatches are recoverable domain errors.

## Roadmap

- [x] Toolchain, presets, cross-platform CI
- [x] N-dimensional Tensor with strides and value semantics
- [x] `const` correctness, ownership rules, `clang-tidy` in CI
- [x] Error-handling policy (`optional` / `expected` / exceptions) and edge-case test suite
- [ ] Generic `Tensor<T>` templates (`float`, `double`, `int32_t`)
- [ ] Computational graph and reverse-mode backpropagation
- [ ] Benchmarks against reference implementations

## License

TBD.