#include <array>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <utility>
#include <vector>

#include "tensor.hpp"

// `assert` se desactiva cuando NDEBUG esta definido (Release), y el test pasaria
// sin comprobar nada. CHECK funciona en cualquier configuracion.
#define CHECK(cond)                                                            \
    do {                                                                       \
        if (!(cond)) {                                                         \
            std::cerr << "\nFALLO: " #cond " (" << __FILE__ << ":" << __LINE__ \
                      << ")\n";                                                \
            std::exit(EXIT_FAILURE);                                           \
        }                                                                      \
    } while (false)

// Indices de un tensor 3D. operator() recibe std::span<const std::size_t>,
// y un std::array se convierte a span automaticamente.
using Idx3 = std::array<std::size_t, 3>;

void test_creacion_y_strides() {
    std::cout << "[TEST] Creacion y calculo de strides... ";

    Tensor t({2, 3, 4});

    CHECK(t.shape().size() == 3);
    CHECK(t.shape()[0] == 2);
    CHECK(t.shape()[1] == 3);
    CHECK(t.shape()[2] == 4);
    CHECK(t.size() == 24);  // 2 * 3 * 4 elementos

    // Row-major: el ultimo eje es contiguo.
    CHECK(t.strides()[2] == 1);
    CHECK(t.strides()[1] == 4);   // saltar una fila = 4 columnas
    CHECK(t.strides()[0] == 12);  // saltar un canal = 3 * 4

    std::cout << "OK\n";
}

void test_acceso_lineal() {
    std::cout << "[TEST] Acceso multidimensional vs indice plano... ";

    Tensor t({2, 3, 4});  // el constructor ya llena con 0.0

    t(Idx3{1, 2, 3}) = 42.0;

    CHECK(t(Idx3{1, 2, 3}) == 42.0);
    // 1*12 + 2*4 + 3*1 = 23 -> ultimo elemento
    CHECK(t.data()[23] == 42.0);

    std::cout << "OK\n";
}

void test_semantica_copia() {
    std::cout << "[TEST] Semantica de copia (deep copy)... ";

    Tensor original({2, 3, 4});
    original(Idx3{0, 0, 0}) = 100.0;

    Tensor copia = original;

    CHECK(copia.shape() == original.shape());
    CHECK(copia.strides() == original.strides());

    // Buffers distintos: la copia es profunda.
    CHECK(copia.data().data() != original.data().data());
    CHECK(copia(Idx3{0, 0, 0}) == 100.0);

    copia(Idx3{0, 0, 0}) = 999.0;
    CHECK(original(Idx3{0, 0, 0}) == 100.0);
    CHECK(copia(Idx3{0, 0, 0}) == 999.0);

    std::cout << "OK\n";
}

void test_semantica_movimiento() {
    std::cout << "[TEST] Semantica de movimiento (robo de recursos)... ";

    Tensor original({2, 3, 4});
    original(Idx3{0, 0, 0}) = 77.0;
    const double* puntero_original = original.data().data();
    const auto strides_original = original.strides();

    Tensor movido = std::move(original);

    // Mover no copia los datos: el nuevo tensor se queda con el mismo buffer.
    CHECK(movido.data().data() == puntero_original);
    CHECK(movido(Idx3{0, 0, 0}) == 77.0);
    CHECK(movido.strides() == strides_original);

    // Un std::vector movido queda vacio en la practica (el estandar solo
    // garantiza "valido pero no especificado").
    CHECK(original.size() == 0);  // NOLINT(bugprone-use-after-move)

    std::cout << "OK\n";
}

void test_tensor_ecalar(){
    std::cout << "[Test] Tensor escalar vacío... ";

    Tensor original(std::vector<std::size_t>{});
    CHECK(original.size() == 1);
    CHECK(original.ndim() == 0);

    std::cout << "Ok\n";
}

void test_tensor_dim_null(){
    std::cout << "[Test] Tensor con dimensión cero... ";

    Tensor original(std::vector<std::size_t>{3, 0});
    CHECK(original.size() == 0);

    std::cout << "Ok\n";
}

void test_tensor_strides(){
    std::cout << "[Test] Tensor - Dimensiones unitarias y strides (singleton)... ";
    Tensor original(std::vector<std::size_t>{1, 5, 1});
    const std::vector<std::size_t> strides_esperados{5, 1, 1};
    
    CHECK(original.size() == 5);
    CHECK(original.strides() == strides_esperados);

    std::cout << "OK\n";
}

void test_excepciones_at(){
    std::cout << "[Test] Excepciones en at()... ";
    Tensor t({2, 3});

    bool lanzo_invalid_agument = false;
    try{
        const std::array<std::size_t, 3> coords_mal{0, 0 ,0};
        static_cast<void>(t.at(coords_mal));
    } catch (const std::invalid_argument&) {
        lanzo_invalid_agument = true;
    }
    CHECK(lanzo_invalid_agument);

    bool lanzo_out_of_range = false;
    try{
        const std::array<std::size_t, 2> coords_fuera{2, 0};
        static_cast<void>(t.at(coords_fuera));
    } catch (const std::out_of_range&){
        lanzo_out_of_range = true;
    }
    CHECK(lanzo_out_of_range);

    std::cout << "Ok\n";
}

int main() {
    std::cout << "=== Ejecutando suite de pruebas de Tensor ===\n";
    test_creacion_y_strides();
    test_acceso_lineal();
    test_semantica_copia();
    test_semantica_movimiento();
    test_tensor_ecalar();
    test_tensor_dim_null();
    test_tensor_strides();
    test_excepciones_at();

    std::cout << "=== Todos los tests pasaron ===\n";
    return EXIT_SUCCESS;
}