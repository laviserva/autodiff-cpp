# Reglas de Propiedad y Paso de Parámetros (Ownership)

Este documento define la política de propiedad de memoria y paso de argumentos para el proyecto `micrograd_cpp` (C++20). El objetivo es evitar copias innecesarias, eliminar ambigüedades sobre la vida útil de los datos y maximizar el rendimiento del procesador sin recurrir a gestión manual de memoria en crudo.

---

## 1. El Principio Fundamental

> **Cada recurso dinámico (heap) tiene exactamente un objeto propietario bien definido (RAII). Las vistas e inspectores nunca son dueños de la memoria.**

En el proyecto:
* `Tensor` es el único dueño de su búfer subyacente (`storage_`).
* Las vistas (`std::span`) son referencias no propietarias a segmentos de memoria contigua. La persona o función que crea un `std::span` debe garantizar que el `Tensor` propietario viva mientras se use la vista.

---

## 2. Matriz de Decisión para Parámetros de Funciones

| Intención de la función | Tipo de Parámetro | Costo de llamada | Regla de uso en el proyecto |
|---|---|---|---|
| **Inspeccionar datos sin modificar** | `const T&` | Paso de 1 puntero (8 bytes) | Usar para tipos pesados (`const Tensor&`, `const std::vector<std::size_t>&`). |
| **Modificar datos in-place** | `T&` | Paso de 1 puntero (8 bytes) | Usar cuando la función transforme el objeto directamente (ej. normalización, relleno). |
| **Consumir o apropiarse del recurso** | `T` (por valor) + `std::move` adentro | Move constructor (0 copias si es rvalue) | Constructores de almacenamiento (ej. `Tensor(std::vector<size_t> shape)`). |
| **Operar sobre secuencias continuas** | `std::span<const double>` (lectura) o `std::span<double>` (escritura) | Paso por valor (16 bytes: ptr + size) | Operaciones numéricas y reducción (sumas, dot products, kernels matemáticos). |
| **Tipos primitivos o enums** | `T` (por valor directo) | 1 a 8 bytes | `double`, `std::size_t`, `DType`. Nunca pasar primitivos por `const&`. |

---

## 3. Ejemplos Concretos en el Código

### A. Paso por Valor (`T`) con `std::move`
Cuando una función o constructor va a almacenar un objeto en un miembro privado, recibe por valor y hace `std::move`:

```cpp
explicit Tensor(std::vector<std::size_t> shape)
    : shape_(std::move(shape)) { ... }
```
* **Por qué:** Si el llamador pasa un temporal (rvalue como `{2, 3, 4}`), el constructor lo roba directamente sin copiar. Si pasa una variable existente (lvalue), el llamador paga una copia explícita solo si lo desea.

### B. Paso por Referencia Constante (`const T&`)
Cuando la función solo necesita leer metadatos de un objeto grande:

```cpp
void imprimir_dimensiones(const Tensor& t);
```
* **Por qué:** No duplica el `std::vector` de datos ni los strides; solo lee a través de una referencia protegida contra modificaciones.

### C. Paso por Referencia Mutable (`T&`)
Cuando la operación altera el estado del tensor sin crear uno nuevo:

```cpp
void poner_a_cero(Tensor& t);
```
* **Por qué:** Evita la reasignación en el heap; reutiliza el búfer que ya tiene asignado el objeto.

### D. Paso por Vista (`std::span`)
Cuando un algoritmo matemático no necesita saber qué es un `Tensor`, solo necesita una tira continua de números:

```cpp
double calcular_suma(std::span<const double> datos);
```
* **Por qué:** Cero acoplamiento. Funciona con un `Tensor`, con un `std::vector<double>`, o con un arreglo estático `double[100]` sin copiar un solo byte.

---

## 4. Retorno de Funciones

1. **Objetos pesados nuevos:** Devolver por valor (`Tensor`). El compilador aplica NRVO (Named Return Value Optimization) o movimiento implícito; el retorno por valor no copia en C++ moderno.
2. **Miembros internos grandes:** Devolver por referencia constante (`const std::vector<std::size_t>& shape() const noexcept`).
3. **Vistas de búferes:** Devolver `std::span<const double>` o `std::span<double>` por valor (16 bytes).
4. **Tipos primitivos:** Devolver por valor sin `const` (`std::size_t size() const noexcept`).
