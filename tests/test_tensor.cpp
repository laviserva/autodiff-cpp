#include <iostream>
#include <cstddef>

#include "tensor.hpp"

void test_creacion_y_strides(){
    std::cout << "[TEST] Creacion y calculo de strides... ";
    
    Tensor3D t(2, 3, 4);

    assert(t.Getchannels() == 2);
    assert(t.Getfilas() == 3);
    assert(t.Getcolumnas() == 4);
    assert(t.size() == 24); // 8(double) * 3 (channels, f, cols)

    // Strides
    assert(t.Getstride_filas() == 4);
    assert(t.Getstride_channels() == 12);

    std::cout<<"Ok"<<std::endl;
}
void test_acceso_lineal(){
    std::cout << "[TEST] Acceso multidimensional vs indice plano... ";

    Tensor3D t(2, 3, 4);
    t.rellenar(0.0);

    t(1, 2, 3) = 42.0;

    assert(t(1,2,3) == 42.0);
    assert(t.data()[23] == 42.0);

    std::cout<<"Ok"<<std::endl;
}
void test_semantica_copia(){
    std::cout << "[TEST] Semantica de copia (Deep Copy)... ";
    Tensor3D original(2, 3, 4);
    original(0, 0, 0) = 100.0;

    Tensor3D copia = original;

    assert(copia.Getchannels() == original.Getchannels());
    assert(copia.Getstride_channels() == original.Getstride_channels());
    assert(copia.Getstride_filas() == original.Getstride_filas());
    
    assert(copia.data() != original.data());
    assert(copia(0, 0, 0) == 100.0);
    
    copia(0, 0, 0) = 999.0;
    assert(original(0, 0, 0) == 100.0);
    assert(copia(0, 0, 0) == 999.0);
    std::cout << "OK\n";
}
void test_semantica_movimiento() {
    std::cout << "[TEST] Semantica de movimiento (Robo de recursos)... ";
    Tensor3D original(2, 3, 4);
    original(0, 0, 0) = 77.0;
    const double* puntero_original = original.data();
    std::size_t stride_ch_original = original.Getstride_channels();
    
    Tensor3D movido = std::move(original);
    
    assert(movido.data() == puntero_original);
    assert(movido(0, 0, 0) == 77.0);
    assert(movido.Getstride_channels() == stride_ch_original);
    
    assert(original.size() == 0);
    std::cout << "OK\n";
}

int main() {
    std::cout << "=== Ejecutando Suite de Pruebas de Tensor3D ===\n";
    test_creacion_y_strides();
    test_acceso_lineal();
    test_semantica_copia();
    test_semantica_movimiento();
    std::cout << "=== Todos los tests pasaron exitosamente ===\n";
    return 0;
}