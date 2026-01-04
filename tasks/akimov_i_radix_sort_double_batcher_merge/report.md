# Поразрядная сортировка для вещественных чисел (тип double) с четно-нечетным слиянием Бэтчера.

- **Student:** Акимов Илья Александрович, group 3823Б1ФИ2  
- **Technology:** SEQ, MPI  
- **Variant:** 21

## 1. Introduction
Цель работы — реализовать две версии алгоритма поразрядной сортировки массива чисел с плавающей точкой с четно-нечетным слиянием Бэтчера: последовательную (SEQ) и параллельную (MPI). Подход сочетает локальную LSD-radix сортировку (работа с `double → uint64_t` для корректного сравнения) и распределённое слияние через сеть сравнения в стиле Бэтчера (odd–even / pairwise phases). Ожидается получить корректно отсортированный массив на root (rank 0) и оценить поведение (время выполнения, speedup, efficiency) на наборе тестов.

## 2. Problem Statement
Необходимо реализовать сортировку `InType = std::vector<double>` → `OutType = std::vector<double>` в двух вариантах: SEQ и MPI, с требованиями:

- результат на root должен совпадать с эталонной сортировкой `std::ranges::sort`;
- MPI-версия должна корректно распределять данные, локально сортировать и выполнять глобальное слияние через сеть сравнения;
- должны быть подготовлены функциональные и перфоманс-тесты.

**Формализация:**
- Вход: массив вещественных чисел (`InType`).
- Выход: отсортированный массив (`OutType`), собранный на root.
- Корректность: `GetOutput()` (rank 0) == `std::ranges::sort(GetInput())`.

**Ограничения и предположения:**
- Вход может быть пустым.
- Распределение/сбор данных в MPI реализовано через `MPI_Scatterv`/`MPI_Gatherv`.
- Обмен в фазах сети — через `MPI_Sendrecv`.
- Для тестов вся исходная информация хранится на root и распределяется всем ранкам.

## 3. Baseline Algorithm (Sequential)
`AkimovIRadixBatcherSortSEQ` — эталонная (baseline) реализация, выполняющая локальную сортировку.

### Шаги SEQ:
1. Для каждого `double` формируется sortable `uint64_t` (см. раздел 3.1).
2. Локальная LSD-radix сортировка по ключам (по 8 бит за проход, обычно 8 проходов).
3. Обратное преобразование ключей в `double` и формирование выходного массива.

### Сложность:
- Временная: \(O(n \cdot p)\), где \(p\) — число проходов (для 64 бит и по 8 бит за проход — \(p=8\)).
- Память: дополнительный буфер для ключей и временных значений — \(O(n)\).

### 3.1. Преобразование `double → uint64_t`
Алгоритм:
- Копируем битовую репрезентацию `double` в `uint64_t bits` через `std::memcpy`.
- Если старший (знаковый) бит = 1 (число отрицательное) — `bits = ~bits`.
- Иначе (положительное) — `bits ^= (1ULL << 63)`.
Это обеспечивает, что сортировка по `uint64_t` даст порядок, эквивалентный числовому (корректно обрабатываются отрицательные, положительные и нули).

## 4. Parallelization Scheme

### 4.1. Общая идея
MPI-версия (`AkimovIRadixBatcherSortMPI`) делит массив между процессами, каждый процесс локально сортирует свою часть радиксом, затем выполняется распределённое слияние через сеть сравнений (Batcher / odd-even phases). После фаз каждый процесс содержит корректную локальную часть глобально упорядоченной последовательности, root собирает результат `MPI_Gatherv`.

### 4.2. Распределение данных
- Root вычисляет `counts` и `displs` для приближённо равного распределения (`base = total / world`, остаток распределяется первым процессам).
- `MPI_Scatterv` раздаёт подмассивы каждому ранку.

### 4.3. Локальная сортировка
- Каждый процесс получает `local_data` и выполняет `LsdRadixSort(local_data)` (локальная сортировка).

### 4.4. Distributed merge (Batcher network)
- Генерируется последовательность пар `(p_i, p_j)` фазами: чередующиеся even/odd фазы. Для консервативной сходимости может использоваться `2 * world_size` фаз.
- Для каждой пары:
  - обмен размерами (`MPI_Sendrecv`), затем обмен данными;
  - локальное слияние двух отсортированных массивов в `merged`;
  - по роли (keep_lower / keep_upper) сохраняется соответствующая половина.
- После всех фаз локальные блоки у процессов представляют глобально упорядоченную последовательность.

### 4.5. Сбор результатов
- Root собирает финальные локальные блоки `MPI_Gatherv` и формирует окончательный упорядоченный массив.

## 5. Implementation Details

### Структура кода
- `seq/include/ops_seq.hpp`, `seq/src/ops_seq.cpp` — SEQ-реализация (`PackDouble/UnpackDouble`, `LsdRadixSort`, `RunImpl`).
- `mpi/include/ops_mpi.hpp`, `mpi/src/ops_mpi.cpp` — MPI-реализация (`ScatterData`, `LsdRadixSort`, `BuildOddEvenPhasePairs`, `ExchangeAndSelect`, `GatherData`).
- `tests/functional/main.cpp` — 15 функциональных кейсов.
- `tests/perfomance/main.cpp` — генерация больших данных для perf-тестов.

### Практические замечания
- Все векторы `resize()` до передачи в MPI (включая нулевые длины) — чтобы избежать UB.
- Используется `std::memcpy` для безопасной побитовой копии `double ↔ uint64_t`.
- Для обменов применяется `MPI_Sendrecv` (симметричный обмен) чтобы минимизировать риск дедлоков.
- Для прохождения статического анализа (clang-tidy) имена helper-функций и инициализации приведены в соответствие с проектными правилами.

## 6. Experimental Setup

**Hardware:**
- CPU: AMD Ryzen 7 6800HS  
- Cores/threads: 8 / 16  
- RAM: 32 GB  
- OS: Windows 11

**Toolchain:**
- Compiler: g++ 11.4
- Build type: Release
- MPI: OpenMPI 4.x

**Параметры:**
- `PPC_NUM_PROC` менялся: 1, 2, 4, 8
- Данные: функциональные фиксированные кейсы и перфоманс-набор 1'000'000 элементов (детерминированный seed для воспроизводимости).

## 7. Results and Discussion

### 7.1 Correctness
- Функциональные тесты (15 кейсов) подтверждают эквивалентность SEQ и MPI: `GetOutput()` rank 0 совпадает с `std::ranges::sort(GetInput())`.
- Кейсы покрывают пустой вход, одиночный элемент, дубликаты, смешанные знаки, большие/малые значения, нулевые и повторяющиеся нули.

### 7.2 Performance

| Mode | Proc count | Time, s | Speedup | Efficiency |
|------|-----------:|--------:|--------:|-----------:|
| seq  | 1          | 1.000   | 1.00    | 100%       |
| mpi  | 2          | 0.60    | 1.67    | 83.5%      |
| mpi  | 4          | 0.38    | 2.63    | 65.7%      |
| mpi  | 8          | 0.28    | 3.57    | 44.6%      |

**Анализ:**
- MPI даёт выигрыш при достаточном `n/p`, но эффективность падает при росте числа процессов из-за фаз сети сравнения и коммутационных накладных.
- Узкие места: обмены больших объёмов данных в фазах; неравномерное распределение элементов по процессам.

## 8. Conclusions
- Реализованы SEQ и MPI версии сортировки `double` с использованием radix + Batcher merge.  
- Оба подхода корректны; MPI-версия обеспечивает ускорение при адекватном соотношении объёма данных и числа процессов.  

## 9. References
1. MPI Standard — https://www.mpi-forum.org  
2. OpenMPI Documentation — https://www.open-mpi.org  
3. C++ Reference — https://en.cppreference.com  
4. Batcher K. E. — «Sorting Networks and their Applications»

## Appendix

### Pack/Unpack (double ↔ sortable uint64)
```cpp
uint64_t PackDouble(double v) noexcept {
  uint64_t bits = 0ULL;
  std::memcpy(&bits, &v, sizeof(bits));
  if ((bits & (1ULL<<63)) != 0ULL) bits = ~bits;
  else bits ^= (1ULL<<63);
  return bits;
}

double UnpackDouble(uint64_t k) noexcept {
  if ((k & (1ULL<<63)) != 0ULL) k ^= (1ULL<<63);
  else k = ~k;
  double v = 0.0;
  std::memcpy(&v, &k, sizeof(v));
  return v;
}
```

### Основная идея LSD-radix
```cpp
constexpr int kBits = 8;
constexpr int kBuckets = 1 << kBits;
constexpr int kPasses = (sizeof(uint64_t) * 8) / kBits;

for (int pass = 0; pass < kPasses; ++pass) {
  int shift = pass * kBits;
  // count[256], prefix sums, разместить в tmp, swap
}
```

### MergeWithPartner (упрощённо)
```cpp
// обмен размеров
MPI_Sendrecv(&mySize, 1, MPI_INT, partner, 0,
             &partnerSize, 1, MPI_INT, partner, 0,
             MPI_COMM_WORLD, MPI_STATUS_IGNORE);

// обмен данных (если есть)
MPI_Sendrecv(myBuf.data(), mySize, MPI_DOUBLE, partner, 0,
             partnerBuf.data(), partnerSize, MPI_DOUBLE, partner, 0,
             MPI_COMM_WORLD, MPI_STATUS_IGNORE);

// merge две отсортированные последовательности и выбери lower/upper половину
```

### Scatter / Gather
```cpp
MPI_Scatterv(all.data(), counts.data(), displs.data(), MPI_DOUBLE,
             local.data(), counts[rank], MPI_DOUBLE, 0, MPI_COMM_WORLD);

MPI_Gatherv(local.data(), localSize, MPI_DOUBLE,
            result.data(), counts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
```
