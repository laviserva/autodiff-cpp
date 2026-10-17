#pragma once

#include <cstddef>
#include <vector>
#include <type_traits>
#include <span>
#include <utility>

enum class DType{
    Float64,
    Float32
};

[[nodiscard]] constexpr std::size_t dtype_size(DType dtype) noexcept {
    switch(dtype){
        case DType::Float64: return 8;
        case DType::Float32: return 4;
    }
    return 0;
}

static_assert(dtype_size(DType::Float64) == sizeof(double), "Float64 debe medir lo mismo que double");
static_assert(dtype_size(DType::Float32) == sizeof(float), "Float32 debe medir lo mismo que float");

class Tensor{
    private:
        std::vector<std::size_t> shape_;
        std::vector<std::size_t> strides_;
        std::vector<double> storage_;
        DType dtype_{DType::Float64};

        // conversión de indices
        [[nodiscard]] std::size_t flat_index(std::span<const std::size_t> indices) const noexcept{
            std::size_t indice_plano = 0;
            const std::size_t total_indice = indices.size();
            for(std::size_t i = 0; i < total_indice; i++){
                indice_plano += indices[i] * strides_[i];
            }
            return indice_plano;
        }
    
    public:
        Tensor() = default;
        explicit Tensor(std::vector<std::size_t> shape):
        shape_(std::move(shape)){
            if (shape_.empty()){
                return;
            }
            strides_.resize(shape_.size());
            strides_.back() = 1;

            for(std::size_t i = shape_.size() - 1; i > 0; i--){
                strides_[i - 1] = strides_[i] * shape_[i];
            }
            storage_.resize(strides_[0] * shape_[0], 0.0);
        }

        Tensor(const Tensor&) = default;
        Tensor& operator=(const Tensor&) = default;
        Tensor(Tensor&&) noexcept = default;
        Tensor& operator=(Tensor&&) noexcept = default;
        ~Tensor() = default;
        
        [[nodiscard]] double& operator()(std::span<const std::size_t> indices) noexcept{
            return storage_[flat_index(indices)];
        }
        [[nodiscard]] const double& operator() (std::span<const std::size_t> indices) const noexcept{
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

static_assert(std::is_move_constructible_v<Tensor>, "Tensor debe poderse mover");
static_assert(std::is_copy_constructible_v<Tensor>, "Tensor debe poderse copiar");
static_assert(!std::is_convertible_v<std::vector<std::size_t>, Tensor>,
              "El constructor debe ser explicit: un vector no se convierte solo en Tensor");
static_assert(std::is_copy_assignable_v<Tensor>, "Tensor debe permitir asignacion por copia");
static_assert(std::is_nothrow_move_constructible_v<Tensor>, "Tensor debe poderse mover sin lanzar excepciones");
static_assert(std::is_nothrow_move_assignable_v<Tensor>, "Tensor debe poderse asignar por movimiento sin excepciones");