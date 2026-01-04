# Отчёт по лабораторной работе  
## «Решение СЛАУ методом простых итераций с использованием MPI»  

**Дисциплина:** Параллельное программирование  
**Преподаватель:** Нестеров Александр Юрьевич и Оболенский Арсений Андреевич  
**Студент:** Крюков Егор Федорович 3823Б1ФИ1
**Вариант:** 20


## Введение

Решение систем линейных алгебраических уравнений (СЛАУ) является одной из фундаментальных задач вычислительной математики. Метод простых итераций — это итерационный численный метод, который позволяет находить приближённое решение СЛАУ. Параллельные вычисления позволяют ускорить процесс решения больших систем уравнений, что актуально для задач моделирования, обработки данных и машинного обучения.

В рамках данной работы были реализованы последовательная и параллельная версии метода простых итераций для решения СЛАУ.


## Постановка задачи

Задача заключается в решении системы линейных уравнений вида: Ax = b, где A — квадратная матрица коэффициентов размера n, b — вектор правых частей, x — искомый вектор решения.

Метод простых итераций использует итерационную формулу. Итерации продолжаются до достижения заданной точности Eps: x(k+1) - x(k)  < Eps


## Описание алгоритма

Базовый последовательный алгоритм начинается с инициализации начального приближения вектора x нулями. На каждой итерации для каждого i вычисляется новое значение x_i по итерационной формуле. После завершения итераций для всех компонент вычисляется норма разности между новым и старым вектором решения. Если норма меньше заданной точности Eps, процесс останавливается. В противном случае итерации продолжаются до достижения максимального числа итераций.

## Описание схемы параллельного алгоритма

Параллельная реализация с использованием MPI основывается на распределении строк матрицы A и соответствующих компонент векторов b и x между процессами. Каждый процесс вычисляет новые значения для своей части вектора x. После этого с помощью операций MPI (MPI_Allgatherv) обновлённые значения собираются со всех процессов, чтобы каждый процесс имел полный актуальный вектор x для следующей итерации. Норма разности вычисляется локально на каждом процессе для его части, а затем находится глобальная норма. Итерации продолжаются до достижения заданной точности.


## Условия экспериментов

Размеры тестовой матрицы n: 20, 200, 500.
Среда выполнения: Windows, MPI.
Количество процессов MPI: 4, 6 и 8 (задаётся параметром -n при запуске mpiexec).
Измерение времени — средствами тестового окружения GoogleTest (режимы pipeline и task_run).


### Результаты при n = 20

#### 4 процесса

| Режим выполнения | Время (сек) | Комментарий             |
| ---------------- | ----------: | ----------------------- |
| MPI pipeline     |  0.00087096 | Параллельная версия     |
| MPI task_run     |  0.00030684 | Параллельная версия     |
| SEQ pipeline     |  0.00006164 | Последовательная версия |
| SEQ task_run     |  0.00002200 | Последовательная версия |

#### 6 процессов

| Режим выполнения | Время (сек) | Комментарий             |
| ---------------- | ----------: | ----------------------- |
| MPI pipeline     |  0.00086728 | Параллельная версия     |
| MPI task_run     |  0.00147854 | Параллельная версия     |
| SEQ pipeline     |  0.00002384 | Последовательная версия |
| SEQ task_run     |  0.00004000 | Последовательная версия |

#### 8 процессов

| Режим выполнения | Время (сек) | Комментарий             |
| ---------------- | ----------: | ----------------------- |
| MPI pipeline     |  0.00411746 | Параллельная версия     |
| MPI task_run     |  0.00621348 | Параллельная версия     |
| SEQ pipeline     |  0.00002588 | Последовательная версия |
| SEQ task_run     |  0.00003900 | Последовательная версия |

Вывод: при малых размерах матрицы последовательная версия остаётся наиболее эффективной. MPI приводит к росту накладных расходов, что нивелирует выигрыш от параллелизма. И чем больше потоков, тем это заметнее.


### Результаты при n = 200 

#### 4 процесса

| Режим выполнения | Время (сек) | Комментарий             |
| ---------------- | ----------: | ----------------------- |
| MPI pipeline     |  0.00907084 | Параллельная версия     |
| MPI task_run     |  0.00864608 | Параллельная версия     |
| SEQ pipeline     |  0.02082334 | Последовательная версия |
| SEQ task_run     |  0.01980000 | Последовательная версия |

#### 6 процессов

| Режим выполнения | Время (сек) | Комментарий             |
| ---------------- | ----------: | ----------------------- |
| MPI pipeline     |  0.01021396 | Параллельная версия     |
| MPI task_run     |  0.00881760 | Параллельная версия     |
| SEQ pipeline     |  0.02261434 | Последовательная версия |
| SEQ task_run     |  0.02140000 | Последовательная версия |

#### 8 процессов

| Режим выполнения | Время (сек) | Комментарий             |
| ---------------- | ----------: | ----------------------- |
| MPI pipeline     |  0.08793538 | Параллельная версия     |
| MPI task_run     |  0.40329608 | Параллельная версия     |
| SEQ pipeline     |  0.02561020 | Последовательная версия |
| SEQ task_run     |  0.02390000 | Последовательная версия |

Вывод: при n = 200 наилучшие результаты достигаются при использовании 4–6 MPI‑процессов. Тут MPI начинает заметно выигрывать по времени. При 8 процессах заметно проявляются накладные расходы на коммуникации и синхронизацию.

### Результаты при n = 500

#### 4 процесса

| Режим выполнения | Время (сек) | Комментарий             |
| ---------------- | ----------: | ----------------------- |
| MPI pipeline     |  0.11769880 | Параллельная версия     |
| MPI task_run     |  0.12385586 | Параллельная версия     |
| SEQ pipeline     |  0.35500810 | Последовательная версия |
| SEQ task_run     |  0.33300000 | Последовательная версия |

#### 6 процессов

| Режим выполнения | Время (сек) | Комментарий             |
| ---------------- | ----------: | ----------------------- |
| MPI pipeline     |  0.08243434 | Параллельная версия     |
| MPI task_run     |  0.09656232 | Параллельная версия     |
| SEQ pipeline     |  0.35716270 | Последовательная версия |
| SEQ task_run     |  0.33600000 | Последовательная версия |

#### 8 процессов

| Режим выполнения | Время (сек) | Комментарий             |
| ---------------- | ----------: | ----------------------- |
| MPI pipeline     |  3.52576780 | Параллельная версия     |
| MPI task_run     |  3.57389524 | Параллельная версия     |
| SEQ pipeline     |  0.49392714 | Последовательная версия |
| SEQ task_run     |  0.47000000 | Последовательная версия |

Вывод: при больших размерах системы оптимальным является использование 4–6 MPI‑процессов. При 8 процессах наблюдается резкое ухудшение производительности из‑за существенных коммуникационных издержек.


## Заключение

В ходе работы успешно реализованы последовательный и параллельный алгоритмы решения СЛАУ методом простых итераций. Параллельная реализация с использованием MPI показывает значительное ускорение на больших размерах системы, однако на малых размерах накладные расходы на коммуникацию делают последовательную версию более эффективной. Оптимальная область применения параллельного алгоритма — задачи с большими системами уравнений.


## Источники

1. Лекции Сысоева Александра Владимировича
2. Практические занятия Нестерова Александра Юрьевича и Оболенского Арсения Андреевича
3. Документация по курсу
4. Различные источники из интернета


## Приложение. Исходный код

### SEQ-версия

```cpp
namespace krykov_e_simple_iterations {

constexpr double kEps = 1e-5;
constexpr int kMaxIter = 10000;

KrykovESimpleIterationsSEQ::KrykovESimpleIterationsSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

bool KrykovESimpleIterationsSEQ::ValidationImpl() {
  const auto &[n, A, b] = GetInput();
  return n > 0 && A.size() == n * n && b.size() == n;
}

bool KrykovESimpleIterationsSEQ::PreProcessingImpl() {
  return true;
}

bool KrykovESimpleIterationsSEQ::RunImpl() {
  const auto &[n, A, b] = GetInput();

  std::vector<double> x(n, 0.0);
  std::vector<double> x_new(n, 0.0);

  for (int iter = 0; iter < kMaxIter; ++iter) {
    for (size_t i = 0; i < n; ++i) {
      double sum = 0.0;
      for (size_t j = 0; j < n; ++j) {
        if (j != i) {
          sum += A[(i * n) + j] * x[j];
        }
      }
      x_new[i] = (b[i] - sum) / A[(i * n) + i];
    }

    double norm = 0.0;
    for (size_t i = 0; i < n; ++i) {
      double diff = x_new[i] - x[i];
      norm += diff * diff;
    }

    x = x_new;

    if (std::sqrt(norm) < kEps) {
      break;
    }
  }

  GetOutput() = x;
  return true;
}

bool KrykovESimpleIterationsSEQ::PostProcessingImpl() {
  return !GetOutput().empty();
}

}  // namespace krykov_e_simple_iterations

```

### MPI-версия
```cpp
namespace krykov_e_simple_iterations {
namespace {

constexpr double kEps = 1e-5;
constexpr int kMaxIter = 10000;

void CalculateLocalXNew(int start, int count, size_t n, const std::vector<double> &local_a,
                        const std::vector<double> &local_b, const std::vector<double> &x,
                        std::vector<double> &local_x_new) {
  for (int i = 0; i < count; ++i) {
    int global_i = start + i;
    double sum = 0.0;
    for (size_t j = 0; j < n; ++j) {
      if (std::cmp_not_equal(j, global_i)) {
        sum += local_a[i * n + j] * x[j];
      }
    }
    local_x_new[i] = (local_b[i] - sum) / local_a[i * n + global_i];
  }
}

double CalculateLocalNorm(int start, int count, const std::vector<double> &x_new, const std::vector<double> &x) {
  double local_norm = 0.0;
  for (int i = 0; i < count; ++i) {
    int gi = start + i;
    double diff = x_new[gi] - x[gi];
    local_norm += diff * diff;
  }
  return local_norm;
}

void CalculateChunkSizesAndDispls(int size, int n, std::vector<int> &chunk_sizes, std::vector<int> &displs) {
  int base = n / size;
  int rem = n % size;

  displs[0] = 0;
  for (int i = 0; i < size; ++i) {
    chunk_sizes[i] = base + (i < rem ? 1 : 0);
    if (i > 0) {
      displs[i] = displs[i - 1] + chunk_sizes[i - 1];
    }
  }
}

void CalculateMatrixChunkSizesAndDispls(int size, int n, std::vector<int> &matrix_chunk_sizes,
                                        std::vector<int> &matrix_displs) {
  int base = n / size;
  int rem = n % size;

  matrix_displs[0] = 0;
  for (int i = 0; i < size; ++i) {
    int rows = base + (i < rem ? 1 : 0);
    matrix_chunk_sizes[i] = rows * n;
    if (i > 0) {
      matrix_displs[i] = matrix_displs[i - 1] + matrix_chunk_sizes[i - 1];
    }
  }
}

}  // namespace

KrykovESimpleIterationsMPI::KrykovESimpleIterationsMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

bool KrykovESimpleIterationsMPI::ValidationImpl() {
  const auto &[n, a, b] = GetInput();
  return n > 0 && a.size() == n * n && b.size() == n;
}

bool KrykovESimpleIterationsMPI::PreProcessingImpl() {
  return true;
}

bool KrykovESimpleIterationsMPI::RunImpl() {
  int size = 0;
  int rank = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  size_t n = 0;
  std::vector<double> a;
  std::vector<double> b;

  if (rank == 0) {
    const auto &input = GetInput();
    n = std::get<0>(input);
    a = std::get<1>(input);
    b = std::get<2>(input);
  }

  MPI_Bcast(&n, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);

  std::vector<int> chunk_sizes(size);
  std::vector<int> displs(size);
  CalculateChunkSizesAndDispls(size, static_cast<int>(n), chunk_sizes, displs);

  std::vector<int> matrix_chunk_sizes(size);
  std::vector<int> matrix_displs(size);
  CalculateMatrixChunkSizesAndDispls(size, static_cast<int>(n), matrix_chunk_sizes, matrix_displs);

  int local_rows = chunk_sizes[rank];
  int local_matrix_size = matrix_chunk_sizes[rank];

  std::vector<double> local_a(local_matrix_size);
  std::vector<double> local_b(local_rows);

  MPI_Scatterv(rank == 0 ? a.data() : nullptr, matrix_chunk_sizes.data(), matrix_displs.data(), MPI_DOUBLE,
               local_a.data(), local_matrix_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  MPI_Scatterv(rank == 0 ? b.data() : nullptr, chunk_sizes.data(), displs.data(), MPI_DOUBLE, local_b.data(),
               local_rows, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  int start = displs[rank];
  int count = local_rows;

  std::vector<double> x(n, 0.0);
  std::vector<double> x_new(n, 0.0);
  std::vector<double> local_x_new(count, 0.0);

  std::vector<int> recv_counts(size);
  std::vector<int> allgather_displs(size);
  CalculateChunkSizesAndDispls(size, static_cast<int>(n), recv_counts, allgather_displs);

  for (int iter = 0; iter < kMaxIter; ++iter) {
    CalculateLocalXNew(start, count, n, local_a, local_b, x, local_x_new);

    MPI_Allgatherv(local_x_new.data(), count, MPI_DOUBLE, x_new.data(), recv_counts.data(), allgather_displs.data(),
                   MPI_DOUBLE, MPI_COMM_WORLD);

    double local_norm = CalculateLocalNorm(start, count, x_new, x);
    double global_norm = 0.0;
    MPI_Allreduce(&local_norm, &global_norm, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    x = x_new;

    if (std::sqrt(global_norm) < kEps) {
      break;
    }
  }

  GetOutput() = x;

  return true;
}

bool KrykovESimpleIterationsMPI::PostProcessingImpl() {
  return true;
}

}  // namespace krykov_e_simple_iterations


```