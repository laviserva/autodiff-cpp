#pragma once

#include <cassert>
#include <cstddef>
#include <limits>
#include <span>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>
#include <cstdint>

enum class DType {
    Float64,
    Float32,
    Int32
};

[[nodiscard]] constexpr std::size_t dtype_size(DType dtype) noexcept {
    switch (dtype) {
        case DType::Float64: return 8;
        case DType::Float32: return 4;
        case DType::Int32: return 4;
    }
    return 0;
}

static_assert(dtype_size(DType::Float64) == sizeof(double), "Float64 must have the same size as double");
static_assert(dtype_size(DType::Float32) == sizeof(float), "Float32 must have the same size as float");
static_assert(dtype_size(DType::Int32) == sizeof(std::int32_t), "Int32 must have the same size as int32_t");

template <typename T>
[[nodiscard]] constexpr DType dtype_of() noexcept {
    if constexpr (std::is_same_v<T, double> ){
        return DType::Float64;
    } else if constexpr (std::is_same_v<T, float> ) {
        return DType::Float32;
    } else if constexpr (std::is_same_v<T, std::int32_t>) {
        return DType::Int32;
    }
}

template <typename T = double>
class Tensor {
    static_assert(std::is_same_v<T, double> ||
                  std::is_same_v<T, float> ||
                  std::is_same_v<T, std::int32_t>,
                  "Tensor<T> only support float, double or int32_t");
private:
    std::vector<std::size_t> shape_;
    std::vector<std::size_t> strides_;
    std::vector<T> storage_;

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
        storage_.assign(numel, T{});
    }

    Tensor(const Tensor&) = default;
    Tensor& operator=(const Tensor&) = default;
    Tensor(Tensor&&) noexcept = default;
    Tensor& operator=(Tensor&&) noexcept = default;
    ~Tensor() = default;

    // Unchecked access (precondition). Fast.
    [[nodiscard]] T& operator()(std::span<const std::size_t> indices) noexcept {
        return storage_[flat_index(indices)];
    }
    [[nodiscard]] const T& operator()(std::span<const std::size_t> indices) const noexcept {
        return storage_[flat_index(indices)];
    }

    // Checked access: throws invalid_argument or out_of_range.
    [[nodiscard]] T& at(std::span<const std::size_t> indices) {
        check_indices(indices);
        return storage_[flat_index(indices)];
    }
    [[nodiscard]] const T& at(std::span<const std::size_t> indices) const {
        check_indices(indices);
        return storage_[flat_index(indices)];
    }

    [[nodiscard]] const std::vector<std::size_t>& shape() const noexcept { return shape_; }
    [[nodiscard]] const std::vector<std::size_t>& strides() const noexcept { return strides_; }
    [[nodiscard]] std::size_t size() const noexcept { return storage_.size(); }
    [[nodiscard]] std::size_t ndim() const noexcept { return shape_.size(); }
    [[nodiscard]] const std::vector<T>& data() const noexcept { return storage_; }
    [[nodiscard]] DType dtype() const noexcept { return dtype_of<T>(); }
    [[nodiscard]] std::span<const T> span() const noexcept { return storage_; }
    [[nodiscard]] std::span<T> span() noexcept { return storage_; }
};

template <typename T>
constexpr bool tensor_contract_ok() {
    static_assert(std::is_move_constructible_v<Tensor<T>>);
    static_assert(std::is_copy_constructible_v<Tensor<T>>);
    static_assert(std::is_copy_assignable_v<Tensor<T>>);
    static_assert(std::is_nothrow_move_constructible_v<Tensor<T>>);
    static_assert(std::is_nothrow_move_assignable_v<Tensor<T>>);
    static_assert(!std::is_convertible_v<std::vector<std::size_t>, Tensor<T>>,
                  "The constructor must be explicit");
    return true;
}

static_assert(tensor_contract_ok<double>());
static_assert(tensor_contract_ok<float>());
static_assert(tensor_contract_ok<std::int32_t>());