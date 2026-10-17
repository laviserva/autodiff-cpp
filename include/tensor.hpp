#include <iostream>
#include <cstddef>
#include <vector>
#include <algorithm>
#include <cassert>
#include <type_traits>
#include <span>

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
        Tensor(std::vector<std::size_t> shape_):
        shape_(shape_){
            if (shape_.empty()){
                return;
            }
            strides_.resize(shape_.size());
            strides_[shape_.size() - 1] = 1;

            for(std::ptrdiff_t i = static_cast<std::ptrdiff_t>(shape_.size()) - 2; i >= 0; --i){
                strides_[i] = strides_[i + 1] * shape_[i + 1];
            }

            storage_.resize(strides_[0] * shape_[0], 0.0);
        }

        double& operator()(std::span<const std::size_t> shape){
            std::size_t indice_plano = 0;
            std::size_t long_indices = shape.size();
            for (std::size_t i = 0; i < long_indices; i++){
                indice_plano = indice_plano + shape[i] * strides_[i];
            }
            return storage_[indice_plano];
        }
        const double& operator()(std::span<const std::size_t> shape) const {
            std::size_t indice_plano = 0;
            std::size_t long_indices = shape.size();
            for (std::size_t i = 0; i < long_indices; i++){
                indice_plano = indice_plano + shape[i] * strides_[i];
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
static_assert(std::is_copy_constructible_v<Tensor>, "Tensor debe ser copiable");
static_assert(std::is_copy_assignable_v<Tensor>, "Tensor debe permitir asignacion por copia");
static_assert(std::is_nothrow_move_constructible_v<Tensor>, "Tensor debe poderse mover sin lanzar excepciones");
static_assert(std::is_nothrow_move_assignable_v<Tensor>, "Tensor debe poderse asignar por movimiento sin excepciones");