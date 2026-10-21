#pragma once

#include <cassert>
#include <cstddef>
#include <limits>
#include <span>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

enum class DType {
    Float64,
    Float32
};

[[nodiscard]] constexpr std::size_t dtype_size(DType dtype) noexcept {
    switch (dtype) {
        case DType::Float64: return 8;
        case DType::Float32: return 4;
    }
    return 0;
}

static_assert(dtype_size(DType::Float64) == sizeof(double), "Float64 must have the same size as double");
static_assert(dtype_size(DType::Float32) == sizeof(float), "Float32 must have the same size as float");

class Tensor {
private:
    std::vector<std::size_t> shape_;
    std::vector<std::size_t> strides_;
    std::vector<double> storage_;
    DType dtype_{DType::Float64};

    // Precondition (checked only by assert in debug builds): valid indices.
    [[nodiscard]] std::size_t flat_index(std::span<const std::size_t> indices) const noexcept {
        assert(indices.size() == shape_.size() && "Number of indices must equal ndim()");
        std::size_t flat = 0;
        for (std::size_t i = 0; i < indices.size(); ++i) {
            assert(indices[i] < shape_[i] && "Index out of bounds for the given shape");
            flat += indices[i] * strides_[i];
        }
        return flat;
    }

    // Runtime validation, used only by at().
    void check_indices(std::span<const std::size_t> indices) const {
        if (indices.size() != shape_.size()) {
            throw std::invalid_argument("Number of indices does not match ndim()");
        }
        for (std::size_t i = 0; i < indices.size(); ++i) {
            if (indices[i] >= shape_[i]) {
                throw std::out_of_range("Index out of range");
            }
        }
    }

public:
    Tensor() = default;

    // shape {}      -> scalar, size() == 1
    // shape {3, 0}  -> empty,  size() == 0
    // Throws std::overflow_error if the element count does not fit in size_t.
    explicit Tensor(std::vector<std::size_t> shape)
        : shape_(std::move(shape)) {
        strides_.resize(shape_.size());
        std::size_t numel = 1;
        for (std::size_t i = shape_.size(); i-- > 0;) {
            strides_[i] = numel;
            if (shape_[i] != 0 &&
                numel > std::numeric_limits<std::size_t>::max() / shape_[i]) {
                throw std::overflow_error("Tensor too large: overflow computing numel");
            }
            numel *= shape_[i];
        }
        storage_.assign(numel, 0.0);
    }

    Tensor(const Tensor&) = default;
    Tensor& operator=(const Tensor&) = default;
    Tensor(Tensor&&) noexcept = default;
    Tensor& operator=(Tensor&&) noexcept = default;
    ~Tensor() = default;

    // Unchecked access (precondition). Fast.
    [[nodiscard]] double& operator()(std::span<const std::size_t> indices) noexcept {
        return storage_[flat_index(indices)];
    }
    [[nodiscard]] const double& operator()(std::span<const std::size_t> indices) const noexcept {
        return storage_[flat_index(indices)];
    }

    // Checked access: throws invalid_argument or out_of_range.
    [[nodiscard]] double& at(std::span<const std::size_t> indices) {
        check_indices(indices);
        return storage_[flat_index(indices)];
    }
    [[nodiscard]] const double& at(std::span<const std::size_t> indices) const {
        check_indices(indices);
        return storage_[flat_index(indices)];
    }

    [[nodiscard]] const std::vector<std::size_t>& shape() const noexcept { return shape_; }
    [[nodiscard]] const std::vector<std::size_t>& strides() const noexcept { return strides_; }
    [[nodiscard]] std::size_t size() const noexcept { return storage_.size(); }
    [[nodiscard]] std::size_t ndim() const noexcept { return shape_.size(); }
    [[nodiscard]] const std::vector<double>& data() const noexcept { return storage_; }
    [[nodiscard]] DType dtype() const noexcept { return dtype_; }
    [[nodiscard]] std::span<const double> span() const noexcept { return storage_; }
    [[nodiscard]] std::span<double> span() noexcept { return storage_; }
};

static_assert(std::is_move_constructible_v<Tensor>, "Tensor must be move constructible");
static_assert(std::is_copy_constructible_v<Tensor>, "Tensor must be copy constructible");
static_assert(!std::is_convertible_v<std::vector<std::size_t>, Tensor>,
              "The constructor must be explicit: a vector must not implicitly convert to a Tensor");
static_assert(std::is_copy_assignable_v<Tensor>, "Tensor must be copy assignable");
static_assert(std::is_nothrow_move_constructible_v<Tensor>, "Tensor must be nothrow move constructible");
static_assert(std::is_nothrow_move_assignable_v<Tensor>, "Tensor must be nothrow move assignable");