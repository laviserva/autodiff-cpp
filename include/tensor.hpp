#include <iostream>
#include <cstddef>
#include <vector>
#include <algorithm>
#include <cassert>
#include <type_traits>

class Tensor3D{
    private:
        std::size_t channels {0};
        std::size_t filas {0};
        std::size_t columnas {0};
        std::size_t stride_channels {0};
        std::size_t stride_filas {0};
        std::vector<double> datos;
    public:
        Tensor3D() = default;
        Tensor3D(std::size_t ch, std::size_t f, std::size_t col):
        channels(ch), filas(f), columnas(col),
        stride_channels(f * col), stride_filas(col),
        datos(ch * f * col){
        }
        
        // operador de acceso
        double& operator() (std::size_t ch, std::size_t f, std::size_t col){
            std::size_t index = (ch * stride_channels) + (f * stride_filas) + col;
            return datos[index];
        }

        const double& operator() (std::size_t ch, std::size_t f, std::size_t col) const{
            std::size_t index = (ch * stride_channels) + (f * stride_filas) + col;
            return datos[index];
        }

        void rellenar(double valor){
            std::fill(datos.begin(), datos.end(), valor);
        }

        [[nodiscard]] std::size_t Getchannels() const { return channels; }
        [[nodiscard]] std::size_t Getfilas() const { return filas; }
        [[nodiscard]] std::size_t Getcolumnas() const { return columnas; }
        [[nodiscard]] std::size_t Getstride_channels() const { return stride_channels; }
        [[nodiscard]] std::size_t Getstride_filas() const { return stride_filas; }
        [[nodiscard]] std::size_t size() const { return datos.size(); }
        [[nodiscard]] const double* data() const { return datos.data(); }
};

static_assert(std::is_move_constructible_v<Tensor3D>, "Tensor3D debe poderse mover");
static_assert(std::is_copy_constructible_v<Tensor3D>, "Tensor3D debe poderse copiar");
static_assert(std::is_copy_constructible_v<Tensor3D>, "Tensor3D debe ser copiable");
static_assert(std::is_copy_assignable_v<Tensor3D>, "Tensor3D debe permitir asignacion por copia");
static_assert(std::is_nothrow_move_constructible_v<Tensor3D>, "Tensor3D debe poderse mover sin lanzar excepciones");
static_assert(std::is_nothrow_move_assignable_v<Tensor3D>, "Tensor3D debe poderse asignar por movimiento sin excepciones");