# Умножение плотных матриц. Элементы типа double. Алгоритм Штрассена.

- Студент: Санников Иван Михайлович, Группа: 3823Б1ФИ2
- Технология: SEQ, MPI
- Вариант: 3
## 1. Введение

Умножение матриц — базовая операция в численных методах, машинном обучении и компьютерной графике. При больших размерах матриц стандартный алгоритм становится дорогим, поэтому используют более быстрые методы, например алгоритм Штрассена, который уменьшает число умножений блоков (7 вместо 8 на каждом уровне разбиения).Цель работы — реализовать параллельную версию алгоритма Штрассена на MPI и сравнить её с последовательной реализацией.

## 2. Постановка задачи
Входные данные: std::tuple<std::vector<std::vector<double>>, std::vector<std::vector<double>>> - кортеж матриц, которые мы будем перемножать.

Выходные данные: std::vector<std::vector<double>> - матрица произведения исходных матриц:

## 3. Базовый алгоритм(seq)
Последовательный алгорритм:
- На вход подаются две квадратные матрицы одинакового размера.
- Если размер матрицы не является степенью двойки, матрицы дополняются нулями до ближайшей степени двойки.
- Матрицы преобразуются в линейный массив для удобства работы с памятью.
- Алгоритм выполняется итеративно с использованием собственного стека, который имитирует рекурсивные вызовы алгоритма Штрассена.
- На каждом шаге, если размер текущей подматрицы меньше или равен пороговому значению, используется классическое умножение матриц.
- В противном случае вычисляются семь произведений по формуле Штрасссена с использованием операций сложения и вычитания подматриц.
- Полученные промежуточные результаты комбинируются в итоговую матрицу.
- После завершения вычислений результат обрезается до исходного размера входных матриц.

```cpp
bool SannikovIShtrassenAlgorithmSEQ::RunImpl() {
  (void)this;
  const auto& input = GetInput();
  const auto& a_in = std::get<0>(input);
  const auto& b_in = std::get<1>(input);

  const std::size_t n0 = a_in.size();
  const std::size_t m = NextPow2(n0);
  Flat a_flat;
  Flat b_flat;
  PackPadToFlat(a_in, n0, m, &a_flat);
  PackPadToFlat(b_in, n0, m, &b_flat);
  const View a0 = MakeView(a_flat, m);
  const View b0 = MakeView(b_flat, m);
  const Flat c_flat = Shtrassen(a0, b0);
  GetOutput() = UnpackCropFromFlat(c_flat, m, n0);

  return !GetOutput().empty();
}
```

## 4. Описание параллельного алгоритма 

1. Получаем входные данные: две квадратные матрицы одинакового размера из std::vector<std::vector<double>>.
2. На процессе с rank = 0 определяется исходный размер матриц и вычисляется ближайшая степень двойки. Это значение рассылается всем процессам с помощью MPI_Bcast.
3. Матрицы дополняются нулями до размера m × m и преобразуются в линейный массив (Flat) для удобства передачи данных и вычислений.
4. Полные дополненные матрицы рассылаются всем процессам при помощи MPI_Bcast.
5. Матрицы логически разбиваются на подматрицы, необходимые для вычисления семи произведений алгоритма Штрассена.
6. Каждому процессу назначается одна или несколько задач вычисления промежуточных матриц M1–M7 в зависимости от номера процесса (rank).
7. Каждый процесс вычисляет только свои локальные части результата с использованием классического умножения матриц.
8. Все промежуточные результаты собираются на корневом процессе с помощью операций MPI_Reduce.
9. На корневом процессе из промежуточных матриц формируется итоговая матрица результата по формулам алгоритма Штрассена.
10. Итоговая матрица обрезается до исходного размера и рассылается всем процессам.

## 5. Experimental Setup

- Hardware/OS: Intel i9 13900KF, 24 ядра, RAM: 32Gb, OS: Windows 11
- Toolchain: Cmake 3.28.3, g++ (Ubuntu 14.2.0 x86_64), Docker-контейнер, Режим сборки: Release.
- Data: Для тестов на производительность использовалась матрица A размером 256 на 256 заполненнаная по алгоритму: элемент i строки j столбца равен (i * 13 + j * 7) % 100, матрица B размером 256 на 256 заполненнаная по алгоритму: элемент i строки j столбца равен (i * 5 + j * 11) % 100.

## 6. Результаты

### 6.1 Корректность 

Корректность работы алгоритма проверена при помощи технологии Google Test. Для проверки использовался набор параметризованных тестов, в которых входными данными являлись пары квадратных матриц типа double различных размеров и структур. 

### 6.2 Производительность

Входные данные: Матрицы 256 на 256.

| Mode        | Count | Time, s | Speedup | Efficiency |
|-------------|-------|---------|---------|------------|
| seq         | 1     | 0.126   | 1.00    | N/A        |
| mpi         | 2     | 0.112   | 1.09    | 54.6%      |
| mpi         | 4     | 0.056   | 2.19    | 54.8%      |
| mpi         | 8     | 0.032   | 4.06    | 50.7%      |

## 7. Выводы

Из результатов тестов на производительность можно сделать вывод, что данная задача хорошо подходит для реализации на MPI. Алгоритм Штрассена дает 7 независимых тяжелых задач, которые хорошо делятся на процессы. Скрость отчетлива видна на 8 процессах, где время выполнения в 4 рраза мегьше чем на последовательном алгоритме. При этом эффективность алгоритма на разном количестве процессов держится в районе 50%.

## 8. Литература
1. Open MPI: Documentation - https://www.open-mpi.org/doc/
2. Александер Сысоев. Курс лекций по параллельному программированию. Лекция № 3 - https://cloud.unn.ru/s/o5y2jGbxb7XpBJa
3. Parallel Programming 2025-2026 - https://disk.yandex.ru/d/NvHFyhOJCQU65w
4. Matrix Multiplication - https://www.geeksforgeeks.org/dsa/strassens-matrix-multiplication/
5. Александр Охотин. Математические основы алгоритмов, осень 2025 г. Лекция 5. Структуры данных в алгоритме Дейкстры. Пути между всеми парами вершин в графе: алгоритм Варшалла. Пути между всеми парами вершин в графе с весами: алгоритм Флойда–Варшалла. Достижимость в графе и умножение матриц. Алгоритм Штрассена быстрого умножения матриц - https://users.math-cs.spbu.ru/~okhotin/teaching/algorithms1_2025/okhotin_algorithms1_2025_l5.pdf

## 9. Приложение

```cpp

bool SannikovIShtrassenAlgorithmMPI::RunImpl() {
  const auto &input = GetInput();
  const auto &mat_a = std::get<0>(input);
  const auto &mat_b = std::get<1>(input);

  int rank = 0;
  int comm_size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &comm_size);

  std::uint64_t n0_64 = 0U;
  if (rank == 0) {
    n0_64 = static_cast<std::uint64_t>(mat_a.size());
  }
  MPI_Bcast(&n0_64, 1, MPI_UINT64_T, 0, MPI_COMM_WORLD);

  if (!SizeOkU64(n0_64)) {
    return false;
  }

  const int n0 = static_cast<int>(n0_64);

  const auto m_sz = NextPow2(static_cast<std::size_t>(n0));
  if (m_sz > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
    return false;
  }
  const int m = static_cast<int>(m_sz);

  if (m == 1) {
    Matrix out;
    (void)FastPath1x1(rank, mat_a, mat_b, &out);
    GetOutput() = out;
    return true;
  }

  Flat a_full;
  Flat b_full;
  BroadcastFullPads(rank, m, mat_a, mat_b, n0, &a_full, &b_full);

  const int k = m / 2;

  Flat a11;
  Flat a12;
  Flat a21;
  Flat a22;
  Flat b11;
  Flat b12;
  Flat b21;
  Flat b22;
  SplitBlocks(a_full, b_full, m, k, &a11, &a12, &a21, &a22, &b11, &b12, &b21, &b22);

  const auto kk = static_cast<std::size_t>(k) * static_cast<std::size_t>(k);

  Flat m1_loc(kk, 0.0);
  Flat m2_loc(kk, 0.0);
  Flat m3_loc(kk, 0.0);
  Flat m4_loc(kk, 0.0);
  Flat m5_loc(kk, 0.0);
  Flat m6_loc(kk, 0.0);
  Flat m7_loc(kk, 0.0);

  ComputeLocalMi(rank, comm_size, k, a11, a12, a21, a22, b11, b12, b21, b22, &m1_loc, &m2_loc, &m3_loc, &m4_loc,
                 &m5_loc, &m6_loc, &m7_loc);

  Flat m1(kk, 0.0);
  Flat m2(kk, 0.0);
  Flat m3(kk, 0.0);
  Flat m4(kk, 0.0);
  Flat m5(kk, 0.0);
  Flat m6(kk, 0.0);
  Flat m7(kk, 0.0);

  ReduceMiToRoot(kk, m1_loc, m2_loc, m3_loc, m4_loc, m5_loc, m6_loc, m7_loc, &m1, &m2, &m3, &m4, &m5, &m6, &m7);

  Flat c_full;
  if (rank == 0) {
    AssembleCOnRoot(m1, m2, m3, m4, m5, m6, m7, m, k, &c_full);
  }

  Flat c_crop_flat;
  RootCropToFlat(rank, m, n0, c_full, &c_crop_flat);
  MPI_Bcast(c_crop_flat.data(), static_cast<int>(c_crop_flat.size()), MPI_DOUBLE, 0, MPI_COMM_WORLD);

  GetOutput() = FlatToMatrix(n0, c_crop_flat);
  return !GetOutput().empty();
}

```
