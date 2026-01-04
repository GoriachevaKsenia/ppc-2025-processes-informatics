# Отчёт по лабораторной работе  
## «Параллельное нахождение решения СЛУ методом Гаусса (ленточная горизонтальная схема) с использованием MPI»

**Дисциплина:** Параллельное программирование  
**Преподаватель:** Нестеров Александр Юрьевич и Оболенский Арсений Андреевич  
**Студент:** Косолапов Виталий Андреевич 3823Б1ФИ1
**Вариант:** 15
---

## Введение

Решение систем линейных уравнений является одной из базовых задач вычислительной математики. Метод Гаусса с выбором главного элемента по столбцу (схема ленточного горизонтального распределения) является классическим алгоритмом для решения таких систем. Однако при увеличении размерности матрицы время работы последовательного алгоритма становится неприемлемым для практических приложений.

Параллельные вычисления с использованием MPI (Message Passing Interface) позволяют значительно ускорить решение больших систем за счёт распределения вычислений между несколькими процессами. В данной работе реализованы последовательная и параллельная версии метода Гаусса, проведено их сравнение по времени выполнения и корректности.

---

## Постановка задачи

Необходимо реализовать нахождение **вектора решений СЛУ методом Гаусса**.  

Дана система линейных уравнений:

\[
A \cdot x = b
\]

где:
- \( A \) — квадратная матрица коэффициентов размером \( n \times n \);
- \( b \) — вектор правых частей;
- \( x \) — искомый вектор неизвестных.
Нужно найти решение представленное в виде вектора \(x)\.

### Требуется:
1. Реализовать последовательный алгоритм метода Гаусса с выбором главного элемента по столбцу.
2. Реализовать параллельную версию алгоритма с использованием MPI.
3. Провести тестирование на корректность с использованием различных тестовых данных.
4. Сравнить производительность последовательной и параллельной реализаций.

---

## Описание алгоритма

Алгоритм поиска максимального значения по столбцам матрицы можно описать следующим образом:

1. **Прямой ход:**
   - Для каждого шага \( i \) от 0 до \( n-1 \):
     - Выбор ведущего элемента в столбце \( i \), начиная с строки \( i \).
     - Перестановка строк и столбцов для перемещения ведущего элемента на диагональ.
     - Нормировка строки \( i \) делением на ведущий элемент.
     - Вычитание строки \( i \) из последующих строк для обнуления элементов ниже диагонали.
2. **Обратный ход:**
   - Вычисление неизвестных, начиная с последнего уравнения.
3. Вернуть получившийся вектор в качестве резултата.

---

## Описание схемы параллельного алгоритма

В параллельной версии с использованием **MPI**:

1. **Инициализация MPI** — каждый процесс получает свой ранг `rank` и общее число процессов `processes_count`.  
2. **Разделение данных:**  
   - Строки матрицы равномерно распределяются между процессами.  
   - Если количество строк не делится нацело, лишние элементы распределяются по первым процессам.
   - Каждый процесс хранит свой блок строк и соответствующие части вектора правых частей.  
3. **Прямой ход (параллельный):**
   - На каждом шаге выбирается ведущий элемент среди строк, принадлежащих процессу-владельцу текущей строки.
   - Информация о ведущем элементе рассылается всем процессам.
   - Выполняется перестановка столбцов и нормировка строки.
   - Ведущая строка рассылается всем процессам для выполнения вычитания.
4. **Обратный ход (параллельный):**
   - Вычисление неизвестных выполняется с использованием уже рассчитанных значений.
   - Результаты собираются на процессе с рангом 0.
5. **Корректировка ответа:**
   - На процессе с рангом 0 востанавливается порядок элементов в векторе ответа с использование вектора перестановок `col_order`.
5. **Вывод результата и завершение работы MPI.**

---

## Описание программной реализации (MPI-версия)

Реализация выполнена на языке **C++17** с использованием **библиотеки MPI**.  
Класс `KosolapovVGaussMethodTapeHorSchemeMPI` реализует интерфейс решения СЛУ методом Гаусса.  

Основные этапы:
- Инициализация и валидация входных данных;  
- Определение диапазона элементов, обрабатываемых каждым процессом;  
- Передача строк и соответсвующих правых частей с которыми работает каждый процесс при помощи `MPI_Send` и `MPI_Recv`;
- Выбор процесса, который работает с текущей строкой;
- Локальный выбор ведущего элемента и нормализация строки, на выбранном процессе; 
- Рассылка информации о ведущем элементе после его выбора;
- Локальная перестановка столбцов;
- Рассылка ведущей строки и вычитание её из всех последующих строк локально на процессах;
- Выбор процесса, который работает с текущей строкой;
- Вычисление неизвестных на выбранном процессе;
- Рассылка частичных результатов при помощи `MPI_Bcast`;
- Корректировка итогового результата на главном процессе;  
- Возврат итогового значения.

### Ключевые функции:
- `MPI_Comm_rank`, `MPI_Comm_size` — определяют номер процесса и общее количество процессов;  
- `MPI_Send` — отправляет сообщение(данные) другому процессу;
- `MPI_Recv` — принимает сообщение(данные) от другого процесса;
- `MPI_Allreduce` — коллективная операция, которая выполняет редукцию (суммирование, максимум и т.д.) над данными всех процессов и рассылает результат всем процессам;
- `MPI_Bcast` - рассылает данные от одного процесса всем остальным процессам.
- `MPI_Barrier` — синхронизация процессов.

### Валидация данных
Не допускаются матрицы с разными длинами строк.
Не допускаются системы, где размер правой части не совпадает с числом строк в матрице.
Не допускаются пустые матрицы.

---

## Результаты экспериментов

### Условия экспериментов
- Размеры матриц: \(\(500\)*\(500\)\), \(\(10^3\)*\(10^3\)\) элементов.  
- Среда выполнения: Windows, MPI (4 процесса).  
- Измерение времени проводилось встроенными средствами тестового фреймворка GoogleTest.  

---

### Результаты при размере матрицы \(\(500\)*\(500\)\)


| Режим выполнения| Кол-во процессов | Время (сек) | Ускорение | Эффективность |
|-----------------|------------------|-------------|-----------|---------------|
| SEQ pipeline    | 1                | 0.49115     | 1.0       | N/A           |
| SEQ task_run    | 1                | 0.48977     | 1.0       | N/A           |
| MPI pipeline    | 4                | 0.15015     | 3.27      | 81.75%        |
| MPI task_run    | 4                | 0.14871     | 3.29      | 82.33%        |
| MPI pipeline    | 6                | 0.12474     | 3.94      | 65.62%        |
| MPI task_run    | 6                | 0.11839     | 4.14      | 68.95%        |

---

### Результаты при размере матрицы \(\(\(10^3\)\)*\(\(10^3\)\)\)


| Режим выполнения| Кол-во процессов | Время (сек) | Ускорение | Эффективность |
|-----------------|------------------|-------------|-----------|---------------|
| SEQ pipeline    | 1                | 3.8442      | 1.0       | N/A           |
| SEQ task_run    | 1                | 3.8722      | 1.0       | N/A           |
| MPI pipeline    | 4                | 1.1792      | 3.26      | 81.5%        |
| MPI task_run    | 4                | 1.1546      | 3.35      | 83.84%        |

---

## Подтверждение корректности
Результаты последовательной и параллельной версий совпадают для всех протестированных вариаций матриц, включая граничные случаи(единичная матрица, верхнетреугольная и нижнетреугольная матрица). Ошибок не наблюдалось, при тестировании возвращался ожидаемый результат. Таким образом, реализация корректна.

---

## Выводы
1. Реализованы две версии алгоритма — последовательная и MPI.  
2. Параллельная реализация успешно масштабируется и даёт незначительный прирост ускорения при увеличении размера данных.
4. Алгоритм корректен и демонстрирует ожидаемое поведение при всех размерах входных данных.  
5. Эффективность увеличивается, при увелечение входных данных, но снижается при увелечение количества процессов, что связано с увелечение накладных расходов.

---

## Заключение
В ходе лабораторной работы был реализован и протестирован метод Гаусса решения СЛУ с использованием технологии MPI. Проведён сравнительный анализ с последовательной версией, подтверждена корректность вычислений, проведены измерения ускорения и эффективности. Полученные результаты демонстрируют эффективность использования параллельных вычислений для ресурсоёмких задач.

---

## Список литературы

1. Параллельные вычисления. MPI Tutorial. — https://mpitutorial.com  
2. Документация по OpenMPI — https://www.open-mpi.org/doc/ 
3. Лекции Сысоева Александра Владимировича
4. Практические занятия Нестерова Александра Юрьевича и Оболенского Арсения Андреевича

---

## Приложение. Исходный код

### Последовательная версия

```cpp

#include "kosolapov_v_gauss_method_tape_hor_scheme/seq/include/ops_seq.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

#include "kosolapov_v_gauss_method_tape_hor_scheme/common/include/common.hpp"

namespace kosolapov_v_gauss_method_tape_hor_scheme {

KosolapovVGaussMethodTapeHorSchemeSEQ::KosolapovVGaussMethodTapeHorSchemeSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = InType(in);
  GetOutput() = {};
}

bool KosolapovVGaussMethodTapeHorSchemeSEQ::ValidationImpl() {
  const auto &input = GetInput();
  if (input.matrix.empty()) {
    return false;
  }
  int n = static_cast<int>(input.matrix.size());
  if (input.matrix.size() != static_cast<size_t>(n)) {
    return false;
  }
  for (const auto &row : input.matrix) {
    if (row.size() != static_cast<size_t>(n)) {
      return false;
    }
  }
  return input.r_side.size() == static_cast<size_t>(n);
}

bool KosolapovVGaussMethodTapeHorSchemeSEQ::PreProcessingImpl() {
  const auto &input = GetInput();
  int n = static_cast<int>(input.matrix.size());
  GetOutput() = std::vector<double>(n, 0.0);
  return true;
}

bool KosolapovVGaussMethodTapeHorSchemeSEQ::RunImpl() {
  const auto &input = GetInput();
  int n = static_cast<int>(input.matrix.size());

  std::vector<std::vector<double>> a = input.matrix;
  std::vector<double> b = input.r_side;
  std::vector<int> col_order(n);
  for (int i = 0; i < n; i++) {
    col_order[i] = i;
  }
  ForwardElimination(a, b, col_order, n);
  std::vector<double> result = BackwardSubstitution(a, b, col_order, n);
  GetOutput() = result;
  return true;
}

bool KosolapovVGaussMethodTapeHorSchemeSEQ::PostProcessingImpl() {
  return true;
}

void KosolapovVGaussMethodTapeHorSchemeSEQ::ForwardElimination(std::vector<std::vector<double>> &a,
                                                               std::vector<double> &b, std::vector<int> &col_order,
                                                               int n) {
  for (int i = 0; i < n; i++) {
    int leading_col = i;
    double max_elem = std::abs(a[i][i]);
    SelectPivot(i, n, a, max_elem, leading_col);
    SwapRows(leading_col, n, i, a, col_order);
    double cur_el = a[i][i];
    for (int j = i; j < n; j++) {
      a[i][j] /= cur_el;
    }
    b[i] /= cur_el;
    RowSub(i, n, a, b);
  }
}
std::vector<double> KosolapovVGaussMethodTapeHorSchemeSEQ::BackwardSubstitution(std::vector<std::vector<double>> &a,
                                                                                std::vector<double> &b,
                                                                                std::vector<int> &col_order, int n) {
  std::vector<double> output(n);
  for (int i = n - 1; i >= 0; --i) {
    output[col_order[i]] = b[i];
    for (int j = i + 1; j < n; ++j) {
      output[col_order[i]] -= a[i][j] * output[col_order[j]];
    }
  }
  return output;
}
void KosolapovVGaussMethodTapeHorSchemeSEQ::SelectPivot(int i, int n, const std::vector<std::vector<double>> &a,
                                                        double &max_elem, int &leading_col) {
  for (int j = i + 1; j < n; j++) {
    if (std::abs(a[i][j]) > max_elem) {
      max_elem = std::abs(a[i][j]);
      leading_col = j;
    }
  }
}
void KosolapovVGaussMethodTapeHorSchemeSEQ::SwapRows(int leading_col, int n, int i, std::vector<std::vector<double>> &a,
                                                     std::vector<int> &col_order) {
  if (leading_col != i) {
    for (int k = 0; k < n; k++) {
      std::swap(a[k][i], a[k][leading_col]);
    }
    std::swap(col_order[i], col_order[leading_col]);
  }
}
void KosolapovVGaussMethodTapeHorSchemeSEQ::RowSub(int i, int n, std::vector<std::vector<double>> &a,
                                                   std::vector<double> &b) {
  for (int k = i + 1; k < n; k++) {
    double ratio = a[k][i];
    for (int j = i; j < n; j++) {
      a[k][j] -= ratio * a[i][j];
    }
    b[k] -= ratio * b[i];
  }
}
}  // namespace kosolapov_v_gauss_method_tape_hor_scheme


```
---
### MPI-версия
```cpp
#include "kosolapov_v_gauss_method_tape_hor_scheme/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

#include "kosolapov_v_gauss_method_tape_hor_scheme/common/include/common.hpp"

namespace kosolapov_v_gauss_method_tape_hor_scheme {

KosolapovVGaussMethodTapeHorSchemeMPI::KosolapovVGaussMethodTapeHorSchemeMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = InType(in);
  GetOutput() = {};
}

bool KosolapovVGaussMethodTapeHorSchemeMPI::ValidationImpl() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if (rank == 0) {
    const auto &input = GetInput();
    if (input.matrix.empty()) {
      return false;
    }
    for (size_t i = 0; i < input.matrix.size(); ++i) {
      if (input.matrix[i].size() != input.matrix.size()) {
        return false;
      }
    }
    if (input.r_side.size() != static_cast<size_t>(input.matrix.size())) {
      return false;
    }
  }
  return true;
}

bool KosolapovVGaussMethodTapeHorSchemeMPI::PreProcessingImpl() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank == 0) {
    const auto &input = GetInput();
    GetOutput() = std::vector<double>(input.matrix.size(), 0.0);
  } else {
    GetOutput() = std::vector<double>();
  }
  return true;
}

bool KosolapovVGaussMethodTapeHorSchemeMPI::RunImpl() {
  int processes_count = 0;
  int rank = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &processes_count);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  int rows = 0;
  if (rank == 0) {
    const auto &input = GetInput();
    rows = static_cast<int>(input.matrix.size());
  }
  MPI_Bcast(&rows, 1, MPI_INT, 0, MPI_COMM_WORLD);

  int rows_per_process = rows / processes_count;
  int remainder = rows % processes_count;

  int start_row = (rank * rows_per_process) + std::min(rank, remainder);
  int end_row = start_row + rows_per_process + (rank < remainder ? 1 : 0) - 1;
  int local_rows = end_row - start_row + 1;
  std::vector<std::vector<double>> local_matrix;
  std::vector<double> local_rsd;
  std::vector<int> local_row_indices;
  if (rank == 0) {
    DistributeDataFromRoot(local_matrix, local_rsd, local_row_indices, start_row, local_rows, rows, processes_count,
                           rows_per_process, remainder);
  } else {
    local_matrix.resize(local_rows);
    local_rsd.resize(local_rows);
    local_row_indices.resize(local_rows);
    for (int i = 0; i < local_rows; i++) {
      local_matrix[i].resize(rows);
      MPI_Recv(&local_row_indices[i], 1, MPI_INT, 0, i, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      MPI_Recv(local_matrix[i].data(), rows, MPI_DOUBLE, 0, i, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      MPI_Recv(&local_rsd[i], 1, MPI_DOUBLE, 0, i, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
  }
  std::vector<int> col_order(rows);
  for (int i = 0; i < rows; ++i) {
    col_order[i] = i;
  }
  // Прямой ход метода Гаусса
  if (!ForwardElimination(rank, rows, start_row, end_row, local_rows, local_matrix, col_order, local_rsd,
                          local_row_indices)) {
    return false;
  }
  // ОБРАТНЫЙ ХОД МЕТОДА ГАУССА
  std::vector<double> global_solution =
      BackwardSubstitution(rows, start_row, end_row, rank, local_rows, local_matrix, local_rsd);
  std::vector<double> final_solution(rows);
  for (int i = 0; i < rows; i++) {
    final_solution[col_order[i]] = global_solution[i];
  }
  MPI_Bcast(final_solution.data(), rows, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  GetOutput() = final_solution;
  MPI_Barrier(MPI_COMM_WORLD);
  return true;
}

bool KosolapovVGaussMethodTapeHorSchemeMPI::PostProcessingImpl() {
  return true;
}

void KosolapovVGaussMethodTapeHorSchemeMPI::DistributeDataFromRoot(std::vector<std::vector<double>> &local_matrix,
                                                                   std::vector<double> &local_rsd,
                                                                   std::vector<int> &local_row_indices, int start,
                                                                   int local_rows, int columns, int processes_count,
                                                                   int rows_per_proc, int remainder) {
  const auto &input = GetInput();
  local_matrix.resize(local_rows);
  local_rsd.resize(local_rows);
  local_row_indices.resize(local_rows);

  for (int i = 0; i < local_rows; i++) {
    local_row_indices[i] = start + i;
    local_matrix[i] = input.matrix[start + i];
    local_rsd[i] = input.r_side[start + i];
  }

  for (int proc = 1; proc < processes_count; proc++) {
    const int proc_start = (proc * rows_per_proc) + std::min(proc, remainder);
    const int proc_end = proc_start + rows_per_proc + (proc < remainder ? 1 : 0);
    const int proc_rows_count = proc_end - proc_start;

    for (int i = 0; i < proc_rows_count; i++) {
      int idx = proc_start + i;
      MPI_Send(&idx, 1, MPI_INT, proc, i, MPI_COMM_WORLD);
      MPI_Send(input.matrix[proc_start + i].data(), columns, MPI_DOUBLE, proc, i, MPI_COMM_WORLD);
      MPI_Send(&input.r_side[proc_start + i], 1, MPI_DOUBLE, proc, i, MPI_COMM_WORLD);
    }
  }
}
bool KosolapovVGaussMethodTapeHorSchemeMPI::ForwardElimination(
    int rank, int rows, int start_row, int end_row, int local_rows, std::vector<std::vector<double>> &local_matrix,
    std::vector<int> &col_order, std::vector<double> &local_rsd, std::vector<int> &local_row_indices) {
  for (int step = 0; step < rows; step++) {
    int owner_process = -1;
    if (step >= start_row && step <= end_row) {
      owner_process = rank;
    }
    int global_owner = 0;
    MPI_Allreduce(&owner_process, &global_owner, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    int pivot_row_local_idx = -1;
    double pivot_value = 0.0;
    int pivot_col = step;
    SelectPivot(rank, owner_process, step, start_row, local_rows, rows, local_matrix, pivot_value, pivot_col,
                pivot_row_local_idx);
    MPI_Bcast(&pivot_col, 1, MPI_INT, global_owner, MPI_COMM_WORLD);
    MPI_Bcast(&pivot_value, 1, MPI_DOUBLE, global_owner, MPI_COMM_WORLD);
    MPI_Bcast(&pivot_row_local_idx, 1, MPI_INT, global_owner, MPI_COMM_WORLD);
    if (std::abs(pivot_value) < 1e-12) {
      GetOutput() = std::vector<double>(rows, 0.0);
      MPI_Barrier(MPI_COMM_WORLD);
      return false;
    }
    SwapColumns(pivot_col, step, local_rows, local_matrix, col_order);
    NormalizePivot(rank, owner_process, pivot_row_local_idx, step, rows, pivot_value, local_matrix, local_rsd);
    std::vector<double> pivot_row(rows - step, 0.0);
    double pivot_rhs = 0.0;
    PriparePivot(rank, owner_process, pivot_row_local_idx, step, rows, pivot_row, pivot_rhs, local_matrix, local_rsd);
    MPI_Bcast(pivot_row.data(), rows - step, MPI_DOUBLE, global_owner, MPI_COMM_WORLD);
    MPI_Bcast(&pivot_rhs, 1, MPI_DOUBLE, global_owner, MPI_COMM_WORLD);
    RowSub(local_rows, step, rows, local_matrix, local_rsd, local_row_indices, pivot_row, pivot_rhs);
    MPI_Barrier(MPI_COMM_WORLD);
  }
  return true;
}
std::vector<double> KosolapovVGaussMethodTapeHorSchemeMPI::BackwardSubstitution(
    int rows, int start_row, int end_row, int rank, int local_rows, std::vector<std::vector<double>> &local_matrix,
    std::vector<double> &local_rsd) {
  std::vector<double> global_solution(rows, 0.0);
  for (int step = rows - 1; step >= 0; step--) {
    int owner_process = -1;
    if (step >= start_row && step <= end_row) {
      owner_process = rank;
    }
    int global_owner = 0;
    MPI_Allreduce(&owner_process, &global_owner, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    double x_value = 0.0;
    if (rank == owner_process) {
      int local_idx = step - start_row;
      if (local_idx >= 0 && local_idx < local_rows) {
        x_value = local_rsd[local_idx];
        for (int col = step + 1; col < rows; col++) {
          x_value -= local_matrix[local_idx][col] * global_solution[col];
        }
      }
    }
    MPI_Bcast(&x_value, 1, MPI_DOUBLE, global_owner, MPI_COMM_WORLD);
    global_solution[step] = x_value;
    MPI_Barrier(MPI_COMM_WORLD);
  }
  return global_solution;
}
void KosolapovVGaussMethodTapeHorSchemeMPI::SelectPivot(int rank, int owner_process, int step, int start_row,
                                                        int local_rows, int rows,
                                                        const std::vector<std::vector<double>> &local_matrix,
                                                        double &pivot_value, int &pivot_col, int &pivot_row_local_idx) {
  if (rank == owner_process) {
    int local_idx = step - start_row;
    if (local_idx >= 0 && local_idx < local_rows) {
      double max_val = 0.0;
      int max_col = step;
      for (int col = step; col < rows; col++) {
        double val = std::abs(local_matrix[local_idx][col]);
        if (val > std::abs(max_val)) {
          max_val = local_matrix[local_idx][col];
          max_col = col;
        }
      }
      pivot_value = max_val;
      pivot_col = max_col;
      pivot_row_local_idx = local_idx;
    }
  }
}
void KosolapovVGaussMethodTapeHorSchemeMPI::RowSub(int local_rows, int step, int rows,
                                                   std::vector<std::vector<double>> &local_matrix,
                                                   std::vector<double> &local_rsd,
                                                   const std::vector<int> &local_row_indices,
                                                   const std::vector<double> &pivot_row, double pivot_rhs) {
  for (int i = 0; i < local_rows; i++) {
    int global_row = local_row_indices[i];
    if (global_row > step) {
      double coef = local_matrix[i][step];
      if (std::abs(coef) > 1e-12) {
        for (int j = step; j < rows; ++j) {
          local_matrix[i][j] -= coef * pivot_row[j - step];
        }
        local_rsd[i] -= coef * pivot_rhs;
      }
    }
  }
}
void KosolapovVGaussMethodTapeHorSchemeMPI::SwapColumns(int pivot_col, int step, int local_rows,
                                                        std::vector<std::vector<double>> &local_matrix,
                                                        std::vector<int> &col_order) {
  if (pivot_col != step) {
    for (int i = 0; i < local_rows; i++) {
      std::swap(local_matrix[i][step], local_matrix[i][pivot_col]);
    }
    std::swap(col_order[step], col_order[pivot_col]);
  }
}
void KosolapovVGaussMethodTapeHorSchemeMPI::NormalizePivot(int rank, int owner_process, int pivot_row_local_idx,
                                                           int step, int rows, double pivot_value,
                                                           std::vector<std::vector<double>> &local_matrix,
                                                           std::vector<double> &local_rsd) {
  if (rank == owner_process && pivot_row_local_idx >= 0) {
    for (int j = step; j < rows; j++) {
      local_matrix[pivot_row_local_idx][j] /= pivot_value;
    }
    local_rsd[pivot_row_local_idx] /= pivot_value;
  }
}
void KosolapovVGaussMethodTapeHorSchemeMPI::PriparePivot(int rank, int owner_process, int pivot_row_local_idx, int step,
                                                         int rows, std::vector<double> &pivot_row, double &pivot_rhs,
                                                         const std::vector<std::vector<double>> &local_matrix,
                                                         const std::vector<double> &local_rsd) {
  if (rank == owner_process && pivot_row_local_idx >= 0) {
    for (int j = step; j < rows; j++) {
      pivot_row[j - step] = local_matrix[pivot_row_local_idx][j];
    }
    pivot_rhs = local_rsd[pivot_row_local_idx];
  }
}
}  // namespace kosolapov_v_gauss_method_tape_hor_scheme


```