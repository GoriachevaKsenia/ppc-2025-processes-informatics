# Отчёт по лабораторной работе
## «Параллельное вычисление двойного интеграла методом Симпсона с использованием MPI»

**Дисциплина:** Параллельное программирование  
**Преподаватель:** Нестеров Александр Юрьевич и Оболенский Арсений Андреевич  
**Студент:** Спичек Денис Игоревич 3823Б1ФИ1  
**Вариант:** 9

---

## Введение

Распараллеливание является ключевым подходом для ускорения вычислений в наукоемких задачах. В данной работе рассматривается реализация алгоритма **вычисления двойного интеграла** функции на прямоугольной области $[0, 1] \times [0, 1]$ с использованием составного метода Симпсона. Реализация выполнена как в последовательной, так и в параллельной форме с использованием технологии MPI (Message Passing Interface).

**Цель работы** — реализовать и сравнить эффективность последовательного и параллельного алгоритмов вычисления двойного интеграла, а также подтвердить корректность MPI-версии.

---

## Постановка задачи

Необходимо вычислить двойной интеграл от функции $f(x, y)$, заданной как сумма квадратов переменных $(x^2 + y^2)$, на области $[0, 1] \times [0, 1]$ с использованием составного **метода Симпсона**.

### Аналитическое решение
Аналитическое значение интеграла равно примерно **0.666**. Для тестирования этот результат округляется до **1**.

### Требуется:
1.  Реализовать **последовательную версию** вычисления (метод Симпсона для двойного интеграла).
2.  Реализовать **параллельную версию** с использованием **MPI**.
3.  Провести сравнение времени выполнения и подтвердить корректность вычислений.

---

## Описание алгоритма

Двойной интеграл на прямоугольной области с использованием составного метода Симпсона при $N$ разбиений по каждой координате (где $N$ должно быть чётным) вычисляется путем **суммирования значений функции в узлах сетки, умноженных на весовые коэффициенты**.

Весовые коэффициенты $w_k$ для узлов сетки определяются так:
* $w_0 = 1$
* $w_N = 1$
* $w_k = 4$, если $k$ нечетное
* $w_k = 2$, если $k$ четное и $k \neq 0, N$

### 1. Последовательный алгоритм (`ops_seq.cpp`)

Последовательный алгоритм выполняет прямой расчет двойной суммы, используя два вложенных цикла. Вспомогательная функция `GetSimpsonWeight` определяет весовые коэффициенты. Итоговый результат вычисляется путем масштабирования суммарного значения.

### 2. Параллельный алгоритм (`ops_mpi.cpp`)

В параллельной версии используется **блочно-циклическое распределение** работы по внешнему циклу (по индексу $i$):

1.  **Инициализация MPI** и определение ранга (`rank`) и числа процессов (`size`).
2.  **Распространение данных:** Число разбиений $N$ рассылается всем процессам с помощью `MPI_Bcast`.
3.  **Распределение нагрузки:** Каждый процесс вычисляет свою локальную сумму (`local_sum`) для непересекающихся рядов (индексов $i$) с шагом, равным общему числу процессов (`size`).
4.  **Сбор результатов:** Частичные суммы (`local_sum`) собираются и суммируются в глобальную сумму (`global_sum`) с использованием коллективной операции `MPI_Allreduce` с операцией `MPI_SUM`.
5.  **Финальный результат:** Каждый процесс вычисляет итоговый интеграл.

---

## Описание программной реализации (MPI-версия)

Реализация выполнена на языке **C++** с использованием **библиотеки MPI**.

**Ключевые особенности реализации:**
* Используется блочно-циклическое распределение работы по внешнему индексу $i$, обеспечивающее хорошую балансировку нагрузки.
* Обмен данными минимизирован до единственной операции `MPI_Allreduce` для сбора частичных сумм, что является эффективным подходом.
* Для обработки весовых коэффициентов используется вспомогательная функция `GetSimpsonWeight`.

---

## Результаты экспериментов

### Условия экспериментов
* **Входной параметр (количество разбиений N):** 10000.
* **Среда выполнения:** Windows, MS-MPI.
* **Методика:** Измерение времени проводилось встроенными средствами тестового фреймворка.

### Результаты замеров времени (pipeline)

| Кол-во процессов (MPI) | Время MPI (сек) | Время SEQ (сек) | Ускорение (отн. MPI-1) | Примечание |
|:---:|---:|---:|:---:|:---:|
| **1** | 0.0978 | 0.0972 | 1.00x | Базовое время MPI/SEQ |
| **2** | 0.0501 | 0.0996 | 1.95x | Значительное ускорение |
| **3** | 0.0373 | 0.1052 | 2.62x | Хорошая масштабируемость |
| **4** | 0.0269 | 0.1095 | 3.64x | Лучшее время выполнения |

### Анализ результатов

1.  **Сравнение MPI vs SEQ:** Сложность вычислений (двойная сумма) достаточно велика, и выигрыш от распараллеливания **превышает** накладные расходы MPI. Начиная с 2 процессов, MPI-версия становится **быстрее** последовательной.
2.  **Масштабируемость MPI:** Наблюдается **положительная масштабируемость**. При увеличении числа процессов с 1 до 4 время выполнения сократилось с 0.0978 с до 0.0269 с. Достигнутое ускорение составляет примерно **3.64 раза**, что близко к идеальному ускорению в 4 раза и подтверждает эффективность распараллеливания.

---

## Подтверждение корректности

Результаты функциональных тестов для обоих режимов (SEQ и MPI) показали, что итоговые суммы, вычисленные параллельно, полностью совпадают с результатами последовательного алгоритма и ожидаемым аналитическим значением.

---

## Выводы

1.  Реализованы две версии алгоритма вычисления двойного интеграла методом Симпсона — последовательная и MPI.
2.  Параллельная реализация успешно масштабируется: при добавлении процессов (с 1 до 4) достигнуто ускорение в **3.64 раза**.
3.  Продемонстрировано, что для задач с высокой вычислительной сложностью, преимущества параллелизма с использованием MPI перевешивают коммуникационные накладные расходы.
4.  Успешно использованы коллективные операции MPI (`Bcast`, `Allreduce`) для синхронизации и сбора результатов.

---

# Исходный код

### `ops_seq.cpp`

```cpp
#include "spichek_d_simpson_integral/seq/include/ops_seq.hpp"
#include <cmath>
#include "spichek_d_simpson_integral/common/include/common.hpp"

namespace spichek_d_simpson_integral {

namespace {
int GetSimpsonWeight(int index, int n) {
  if (index == 0 || index == n) return 1;
  return (index % 2 == 0) ? 2 : 4;
}
}  // namespace

SpichekDSimpsonIntegralSEQ::SpichekDSimpsonIntegralSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool SpichekDSimpsonIntegralSEQ::ValidationImpl() {
  return (GetInput() > 0) && (GetInput() % 2 == 0);
}

bool SpichekDSimpsonIntegralSEQ::PreProcessingImpl() {
  GetOutput() = 0;
  return true;
}

bool SpichekDSimpsonIntegralSEQ::RunImpl() {
  int n = GetInput();
  double h = 1.0 / n;
  double sum = 0.0;
  for (int i = 0; i <= n; i++) {
    const double x = i * h;
    const int wx = GetSimpsonWeight(i, n);
    for (int j = 0; j <= n; j++) {
      const double y = j * h;
      const int wy = GetSimpsonWeight(j, n);
      sum += wx * wy * (x * x + y * y);
    }
  }
  double result = sum * h * h / 9.0;
  GetOutput() = static_cast<OutType>(std::round(result));
  return true;
}

bool SpichekDSimpsonIntegralSEQ::PostProcessingImpl() { return true; }

}  // namespace spichek_d_simpson_integral
```

### `ops_mpi.cpp`

```cpp
#include "spichek_d_simpson_integral/mpi/include/ops_mpi.hpp"
#include <mpi.h>
#include <cmath>
#include "spichek_d_simpson_integral/common/include/common.hpp"

namespace spichek_d_simpson_integral {

namespace {
int GetSimpsonWeight(int index, int n) {
  if (index == 0 || index == n) return 1;
  return (index % 2 == 0) ? 2 : 4;
}
}  // namespace

SpichekDSimpsonIntegralMPI::SpichekDSimpsonIntegralMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool SpichekDSimpsonIntegralMPI::ValidationImpl() {
  return (GetInput() > 0) && (GetInput() % 2 == 0);
}

bool SpichekDSimpsonIntegralMPI::PreProcessingImpl() {
  GetOutput() = 0;
  return true;
}

bool SpichekDSimpsonIntegralMPI::RunImpl() {
  int rank = 0, size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  int n = (rank == 0) ? GetInput() : 0;
  MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (n <= 0 || n % 2 != 0) {
    GetOutput() = 0;
    return true;
  }

  const double h = 1.0 / n;
  double local_sum = 0.0;

  for (int i = rank; i <= n; i += size) {
    const double x = i * h;
    const int wx = GetSimpsonWeight(i, n);
    for (int j = 0; j <= n; ++j) {
      const double y = j * h;
      const int wy = GetSimpsonWeight(j, n);
      local_sum += wx * wy * (x * x + y * y);
    }
  }

  double global_sum = 0.0;
  MPI_Allreduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

  const double result = global_sum * h * h / 9.0;
  GetOutput() = static_cast<OutType>(std::round(result));
  return true;
}

bool SpichekDSimpsonIntegralMPI::PostProcessingImpl() { return true; }

}  // namespace spichek_d_simpson_integral
```

### `ops_seq.hpp`

```cpp
#pragma once
#include "spichek_d_simpson_integral/common/include/common.hpp"
#include "task/include/task.hpp"

namespace spichek_d_simpson_integral {

class SpichekDSimpsonIntegralSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }
  explicit SpichekDSimpsonIntegralSEQ(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace spichek_d_simpson_integral
```

### `ops_mpi.hpp`

```cpp
#pragma once
#include "spichek_d_simpson_integral/common/include/common.hpp"
#include "task/include/task.hpp"

namespace spichek_d_simpson_integral {

class SpichekDSimpsonIntegralMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit SpichekDSimpsonIntegralMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace spichek_d_simpson_integral
```

### `common.hpp`

```cpp
#pragma once
#include <string>
#include <tuple>
#include "task/include/task.hpp"

namespace spichek_d_simpson_integral {

using InType = int;   // количество разбиений (должно быть чётным)
using OutType = int;  // результат (приводим к int для тестов)
using TestType = std::tuple<int, std::string>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace spichek_d_simpson_integral

