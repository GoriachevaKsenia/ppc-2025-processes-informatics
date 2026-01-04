# Отчет по реализации алгоритма: Метод Гаусса – ленточная горизонтальная схема

- **Студент**: Киселев Игорь Вячеславович, группа 3823Б1ФИ1  
- **Преподаватели**: Нестеров Александр Юрьевич и Оболенский Арсений Андреевич  
- **Технология**: SEQ | MPI  
- **Вариант**: 15  

## 1. Введение

Целью данной работы является реализация и исследование алгоритма решения системы линейных уравнений методом Гаусса для ленточных матриц с использованием горизонтальной схемы распараллеливания. В рамках работы выполнено сравнение последовательной (SEQ) и параллельной (MPI) реализаций алгоритма.

Метод Гаусса является одним из базовых численных методов решения систем линейных уравнений и широко используется в инженерных и научных вычислениях, и при больших размерах входных данных требует большого времени вычислений в стандартной (seq) реализации. Во многих практических задачах матрицы имеют ленточную структуру, то есть ненулевые элементы сосредоточены вблизи главной диагонали. Это позволяет сократить объём вычислений и используемой памяти.

Использование параллельных вычислений становится актуальным при работе с большими размерностями систем. В данной работе рассматривается возможность ускорения вычислений за счёт применения MPI и распределения строк матрицы между процессами.

## 2. Постановка задачи

Требуется реализовать алгоритм решения системы линейных уравнений вида Ax = b методом Гаусса для ленточной матрицы с горизонтальной схемой распределения данных.

### Входные данные

На вход подаётся кортеж:
- матрица коэффициентов  
  `std::vector<std::vector<double>>`
- вектор правых частей  
  `std::vector<double>`
- ширина полуленты `band` типа `std::size_t`

Матрица предполагается квадратной и ленточной.

### Выходные данные

Вектор решения системы:
`std::vector<double>`

### Ограничения и условия

- Матрица должна быть квадратной.
- Размер матрицы совпадает с размером вектора правых частей.
- Матрица является ленточной с заданной шириной band.
- Диагональные элементы не должны быть равны нулю.
- Реализации SEQ и MPI должны выдавать корректное решение с допустимой погрешностью.

## 3. Базовый алгоритм (SEQ)

Последовательная версия алгоритма реализует классический метод Гаусса с учётом ленточной структуры матрицы.

Алгоритм состоит из двух этапов:
1. Прямой ход — приведение матрицы к верхнетреугольному виду.
2. Обратный ход — вычисление значений неизвестных.

При вычислениях учитываются только элементы, находящиеся в пределах ленты, что уменьшает количество операций.

### Псевдокод
Прямой ход:

for k = 0 .. n-1:
нормализовать строку k (деление на A[k][k])

for i = k+1 .. min(n-1, k + band):
    занулить A[i][k] с использованием строки k

Обратный ход:

for i = n-1 .. 0:
x[i] = b[i]
for j = i+1 .. min(n-1, i + band):

Сложность алгоритма: O(n * band).

## 4. Схема распараллеливания (MPI)

### Распределение данных

1. Процесс с рангом 0:
   - определяет размер системы и ширину ленты;
   - разбивает строки матрицы между процессами;
   - преобразует ленточную матрицу в компактный линейный формат;
   - формирует массивы sendcounts и displs.

2. Данные распределяются между процессами с помощью MPI_Scatterv.

### Обработка

Каждый процесс:
- получает свой набор строк матрицы и соответствующие элементы вектора b;
- участвует в прямом ходе метода Гаусса;
- выполняет локальные вычисления в пределах своей части данных.

Для синхронизации используется MPI_Bcast для передачи опорной строки.

### Сбор результатов

На этапе обратного хода вычисленные значения неизвестных передаются между процессами. Итоговый вектор решения формируется и рассылается всем процессам.

## 5. Детали реализации

### Структура проекта

- tasks/kiselev_i_gauss_method_horizontal_tape_scheme/mpi — MPI-реализация.
- tasks/kiselev_i_gauss_method_horizontal_tape_scheme/seq — последовательная реализация.
- tasks/kiselev_i_gauss_method_horizontal_tape_scheme/common — общие типы данных.
- tests/functional — функциональные тесты.
- tests/performance — тесты производительности.

### Особенности реализации

Ленточное хранение матрицы (MPI):  
Матрица хранится в виде одномерного массива шириной (2 * band + 1) для каждой строки, что снижает потребление памяти.

Распределение строк:
- каждая строка обрабатывается ровно одним процессом;
- количество строк между процессами отличается не более чем на одну.

Прямой ход:
- опорная строка определяется владельцем строки;
- строка передаётся всем процессам через MPI_Bcast.

Обратный ход:
- значения неизвестных вычисляются начиная с последней строки;
- результат синхронизируется между процессами.

Память:
- каждый процесс хранит только свою часть матрицы;
- полная матрица присутствует только у процесса с рангом 0.

## 6. Тестовая конфигурация

### Оборудование

- Процессор: Intel(R) Core(TM) i5-8265U CPU @ 1.60GHz  
- ОЗУ: 8 ГБ  
- ОС: Windows 10 Pro  
- MPI: Microsoft MPI  
- Режим компиляции: Release (/O2)

## 7. Корректность результатов и тестирование

### 7.1 Корректность

Корректность реализации проверена с помощью функциональных тестов:
- сравнение результатов MPI и SEQ версий;
- различные размеры матриц;
- разные значения ширины ленты;
- симметричные и несимметричные матрицы.

Решение считается корректным, если максимальная невязка |Ax - b| меньше заданного порога.

Все функциональные тесты пройдены успешно.

### 7.2 Перфоманс-тесты

Перфоманс-тесты проводились на ленточной матрице большого размера.

| Размер данных | Режим        | Время, s | Ускорение |
|--------------|-------------|----------|-----------|
| 6000         | SEQ         | 0.32     | 1.00      |
| 6000         | MPI (2 proc)| 0.22     | 1.45     |
| 6000         | MPI (4 proc)| 0.18     | 1.78      |


## 8. Заключение

В ходе работы были реализованы последовательная и параллельная версии метода Гаусса для ленточных матриц с горизонтальной схемой распределения данных.

Последовательная версия эффективна для небольших размеров задачи. Параллельная версия позволяет распределить вычисления между процессами, и позволяет добиться отличной производительности на больших матрицах.

Использование ленточной структуры матрицы снижает объём вычислений и потребление памяти. Реализация корректно обрабатывает все предусмотренные входные данные и соответствует требованиям проекта.

## 9. Литература

1. https://learning-process.github.io/parallel_programming_course/ru/
2. https://learn.microsoft.com/ru-ru/message-passing-interface/mpi-bcast-function
3. https://learn.microsoft.com/ru-ru/message-passing-interface/mpi-scatterv-function
4. https://habr.com/ru/articles/121235/

## Приложения (код параллельной реализации)
```
namespace {

void MakePartition(int proc, int num, std::vector<int> &cnt, std::vector<int> &disp) {
  cnt.assign(proc, 0);
  disp.assign(proc, 0);

  const int q_coef = num / proc;
  const int r_coef = num % proc;
  int pos = 0;

  for (int index = 0; index < proc; ++index) {
    cnt[index] = q_coef + (index < r_coef ? 1 : 0);
    disp[index] = pos;
    pos += cnt[index];
  }
}

int OwnerOf(int row, const std::vector<int> &cnt, const std::vector<int> &disp) {
  for (std::size_t proc = 0; proc < cnt.size(); ++proc) {
    if (row >= disp[proc] && row < disp[proc] + cnt[proc]) {
      return static_cast<int>(proc);
    }
  }
  return 0;
}

bool BuildPivotRow(int k_index, int band, int w_coef, int rank, int owner, int row0,
                   const std::vector<double> &a_vector, const std::vector<double> &b_vector, std::vector<double> &pivot,
                   double &rhs_pivot) {
  if (rank != owner) {
    return true;
  }

  const int local_k = k_index - row0;
  const auto row_offset = static_cast<std::size_t>(local_k) * static_cast<std::size_t>(w_coef);
  const double *row = &a_vector[row_offset];

  const double diag = row[band];
  if (diag == 0.0) {
    return false;
  }

  const int right = static_cast<int>(pivot.size()) - 1 + k_index;
  for (int j = k_index; j <= right; ++j) {
    pivot[j - k_index] = row[j - (k_index - band)];
  }

  rhs_pivot = b_vector[static_cast<std::size_t>(local_k)];
  return true;
}

bool ApplyElimination(int k_index, int band, int w_coef, int row0, int local_rows, const std::vector<double> &pivot,
                      double rhs_pivot, std::vector<double> &a_vector, std::vector<double> &b_vector) {
  const double diag = pivot[0];
  if (diag == 0.0) {
    return false;
  }

  const int right = k_index + static_cast<int>(pivot.size()) - 1;
  const int row_begin = std::max(row0, k_index + 1);
  const int row_end = std::min(row0 + local_rows - 1, k_index + band);

  for (int index = row_begin; index <= row_end; ++index) {
    const int local_i = index - row0;
    const auto row_offset = static_cast<std::size_t>(local_i) * static_cast<std::size_t>(w_coef);
    double *row = &a_vector[row_offset];

    const int col = k_index - (index - band);
    if (col < 0 || col >= w_coef) {
      continue;
    }

    const double multiplier = row[col] / diag;
    row[col] = 0.0;

    for (int j = k_index + 1; j <= right; ++j) {
      const int idx = j - (index - band);
      if (idx >= 0 && idx < w_coef) {
        row[idx] -= multiplier * pivot[j - k_index];
      }
    }

    b_vector[static_cast<std::size_t>(local_i)] -= multiplier * rhs_pivot;
  }

  return true;
}

bool EliminateForward(int num, int band, int w_coef, int rank, int row0, int local_rows, const std::vector<int> &cnt,
                      const std::vector<int> &disp, std::vector<double> &a_vector, std::vector<double> &b_vector) {
  for (int k_index = 0; k_index < num; ++k_index) {
    const int owner = OwnerOf(k_index, cnt, disp);
    const int right = std::min(num - 1, k_index + band);
    const int length = right - k_index + 1;

    std::vector<double> pivot(static_cast<std::size_t>(length), 0.0);
    double rhs_pivot = 0.0;

    if (!BuildPivotRow(k_index, band, w_coef, rank, owner, row0, a_vector, b_vector, pivot, rhs_pivot)) {
      return false;
    }

    MPI_Bcast(pivot.data(), length, MPI_DOUBLE, owner, MPI_COMM_WORLD);
    MPI_Bcast(&rhs_pivot, 1, MPI_DOUBLE, owner, MPI_COMM_WORLD);

    if (!ApplyElimination(k_index, band, w_coef, row0, local_rows, pivot, rhs_pivot, a_vector, b_vector)) {
      return false;
    }
  }

  return true;
}

bool EliminateBackward(int num, int band, int w_coef, int rank, int row0, const std::vector<int> &cnt,
                       const std::vector<int> &disp, const std::vector<double> &a_vector,
                       const std::vector<double> &b_vector, std::vector<double> &x_vector) {
  x_vector.assign(num, 0.0);

  for (int k_index = num - 1; k_index >= 0; --k_index) {
    const int owner = OwnerOf(k_index, cnt, disp);
    const int right = std::min(num - 1, k_index + band);

    double val = 0.0;

    if (rank == owner) {
      const int local_k = k_index - row0;
      const double *row = &a_vector[static_cast<std::size_t>(local_k) * static_cast<std::size_t>(w_coef)];

      double sum = 0.0;
      for (int j_index = k_index + 1; j_index <= right; ++j_index) {
        const int idx = j_index - (k_index - band);
        if (idx >= 0 && idx < w_coef) {
          sum += row[idx] * x_vector[j_index];
        }
      }

      const double diag = row[band];
      if (diag == 0.0) {
        return false;
      }

      val = (b_vector[static_cast<std::size_t>(local_k)] - sum) / diag;
    }

    MPI_Bcast(&val, 1, MPI_DOUBLE, owner, MPI_COMM_WORLD);
    x_vector[k_index] = val;
  }

  return true;
}

}  // namespace

bool KiselevITestTaskMPI::RunImpl() {
  const auto &[a_in, b_in, band_in] = GetInput();

  int rank = 0;
  int size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  int num = 0;
  int band = 0;
  if (rank == 0) {
    num = static_cast<int>(a_in.size());
    band = static_cast<int>(band_in);
  }

  MPI_Bcast(&num, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&band, 1, MPI_INT, 0, MPI_COMM_WORLD);

  const int w_coef = (2 * band) + 1;

  std::vector<int> cnt;
  std::vector<int> disp;
  MakePartition(size, num, cnt, disp);

  const int local_rows = cnt[rank];
  const int row0 = disp[rank];

  std::vector<double> a_loc(static_cast<std::size_t>(local_rows) * static_cast<std::size_t>(w_coef), 0.0);
  std::vector<double> b_loc(static_cast<std::size_t>(local_rows), 0.0);

  std::vector<int> cnt_a(size);
  std::vector<int> disp_a(size);
  for (int index = 0; index < size; ++index) {
    cnt_a[index] = cnt[index] * w_coef;
    disp_a[index] = disp[index] * w_coef;
  }

  std::vector<double> send_a;
  std::vector<double> send_b;
  if (rank == 0) {
    send_a.assign(static_cast<std::size_t>(num) * static_cast<std::size_t>(w_coef), 0.0);
    send_b = b_in;

    for (int index = 0; index < num; ++index) {
      for (int j_index = std::max(0, index - band); j_index <= std::min(num - 1, index + band); ++j_index) {
        const auto row_index = static_cast<std::size_t>(index);
        const auto col_index = static_cast<std::size_t>(j_index);
        const auto band_offset = static_cast<std::size_t>(j_index - (index - band));
        const auto linear_index = (row_index * static_cast<std::size_t>(w_coef)) + band_offset;

        send_a[linear_index] = a_in[row_index][col_index];
      }
    }
  }

  MPI_Scatterv(rank == 0 ? send_a.data() : nullptr, cnt_a.data(), disp_a.data(), MPI_DOUBLE, a_loc.data(), cnt_a[rank],
               MPI_DOUBLE, 0, MPI_COMM_WORLD);

  MPI_Scatterv(rank == 0 ? send_b.data() : nullptr, cnt.data(), disp.data(), MPI_DOUBLE, b_loc.data(), local_rows,
               MPI_DOUBLE, 0, MPI_COMM_WORLD);

  if (!EliminateForward(num, band, w_coef, rank, row0, local_rows, cnt, disp, a_loc, b_loc)) {
    return false;
  }

  std::vector<double> x_vector;
  if (!EliminateBackward(num, band, w_coef, rank, row0, cnt, disp, a_loc, b_loc, x_vector)) {
    return false;
  }

  GetOutput().swap(x_vector);
  return true;
}

```