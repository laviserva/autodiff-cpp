#include <iostream>

int main() {
    std::cout << "=== Motor Autodiff C++ ===" << std::endl;

#if defined(_MSC_VER)
    std::cout << "Compilador: Microsoft Visual C++ " << _MSC_VER << std::endl;
#elif defined(__clang__)
    std::cout << "Compilador: Clang " << __clang_version__ << std::endl;
#elif defined(__GNUC__)
    std::cout << "Compilador: GCC " << __GNUC__ << "." << __GNUC_MINOR__ << std::endl;
#else
    std::cout << "Compilador: Desconocido" << std::endl;
#endif

#if defined(NDEBUG)
    std::cout << "Modo: Release" << std::endl;
#else
    std::cout << "Modo: Debug" << std::endl;
#endif

    return 0;
}