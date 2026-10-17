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

class Tensor{
    private:
        std::vector<std::size_t> shape_;
        std::vector<std::size_t> strides_;
        std::vector<double> storage_;
        DType dtype_{DType::Float64};
    public:
        Tensor() = default;
        explicit Tensor(std::vector<std::size_t> shape):
        shape_(std::move(shape)){
            if (shape_.empty()){
                return;
            }
            strides_.resize(shape_.size());
            strides_.back() = 1;

            for (std::size_t i = shape_.size() - 1; i > 0; i--){
                strides_[i - 1] = strides_[i] * shape_[i];
            }
            storage_.resize(strides_[0] * shape_[0], 0.0);
        }

        Tensor(const Tensor&) = default;
        Tensor& operator=(const Tensor&) = default;
        Tensor(Tensor&&) noexcept = default;
        Tensor& operator=(Tensor&&) noexcept = default;
        ~Tensor() = default;

        double& operator()(std::span<const std::size_t> indices){
            std::size_t indice_plano = 0;
            std::size_t long_indices = indices.size();
            for (std::size_t i = 0; i < long_indices; i++){
                indice_plano = indice_plano + indices[i] * strides_[i];
            }
            return storage_[indice_plano];
        }
        const double& operator()(std::span<const std::size_t> indices) const {
            std::size_t indice_plano = 0;
            std::size_t long_indices = indices.size();
            for (std::size_t i = 0; i < long_indices; i++){
                indice_plano = indice_plano + indices[i] * strides_[i];
            }
            return storage_[indice_plano];
        }

        [[nodiscard]] const std::vector<std::size_t>& shape() const { return shape_; }
        [[nodiscard]] const std::vector<std::size_t>& strides() const { return strides_; }
        [[nodiscard]] const std::size_t size() const { return storage_.size(); }
        [[nodiscard]] const std::vector<double>& data() const { return storage_; }
        [[nodiscard]] const DType dtype() const { return dtype_; }
        
};

static_assert(std::is_move_constructible_v<Tensor>, "Tensor debe poderse mover");
static_assert(std::is_copy_constructible_v<Tensor>, "Tensor debe poderse copiar");
static_assert(!std::is_convertible_v<std::vector<std::size_t>, Tensor>,
              "El constructor debe ser explicit: un vector no se convierte solo en Tensor");
static_assert(std::is_copy_assignable_v<Tensor>, "Tensor debe permitir asignacion por copia");
static_assert(std::is_nothrow_move_constructible_v<Tensor>, "Tensor debe poderse mover sin lanzar excepciones");
static_assert(std::is_nothrow_move_assignable_v<Tensor>, "Tensor debe poderse asignar por movimiento sin excepciones");