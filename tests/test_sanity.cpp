#include <cassert>
#include <iostream>

int main() {
    int a = 10;
    int b = 20;
    assert((a + b) == 30);

    std::cout << "[PASS] Test de sanidad completado con exito." << std::endl;
    return 0;
}