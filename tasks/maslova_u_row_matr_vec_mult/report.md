# Ленточная горизонтальная схема - умножение матрицы на вектор

- Student: Маслова Ульяна Александровна, group 3823Б1ФИ2
- Technology: SEQ | MPI
- Variant: 11

## 1. Introduction
Проблема: Перемножение плотных матриц большого размера на вектор в последовательном занимает много времени.
Задача: Ускорить этот процесс с помощью параллельных вычислений на MPI.
Ожидаемый результат: Значительное сокращение времени выполнения по сравнению с последовательной версией.

## 2. Problem Statement
Дана матрица A размера rows X cols и вектор B. Необходимо найти вектор C, где каждый i-ый элемент является скалярным произведением i-ой строки матрицы A на вектор B.
- InPut: Структура Matrix и std::vector<double>.
- OutPut: std::vector<double>.

## 3. Baseline Algorithm (Sequential)
Последовательный алгоритм проходит по каждой строке матрицы и вычисляет сумму произведений элементов строки на соответствующие элементы вектора.

## 4. Parallelization Scheme
Ранг 0 считывает размеры матрицы и транслирует их всем остальным процессам (MPI_Bcast).
Вектор B полностью рассылается всем процессам (MPI_Bcast), так как он нужен для вычисления каждой строки.
Матрица A делится на блоки строк. Каждый процесс получает свою порцию данных с помощью MPI_Scatterv. Каждый процесс локально вычисляет произведение своей части строк на вектор.
Результаты собираются на ранге 0 с помощью MPI_Gatherv.

## 5. Implementation Details
- common: Определяет общие типы данных (InType, OutType).
- seq: Содержит простую последовательную реализацию алгоритма.
- mpi: Содержит параллельную MPI-реализацию алгоритма.
- tests: Включает два набора тестов: functional для проверки корректности и performance для замера скорости.

## 6. Experimental Setup
- Аппаратное обеспечение: AMD Ryzen 7 7840HS (8 ядер, 16 логических процессоров, базовая частота 3,80 ГГц)
- ОЗУ — 16 ГБ 
- Операционная система: Windows 11
- Компилятор: g++
- Тип сборки: Release

## 7. Results and Discussion

### 7.1 Correctness
Корректность проверялась на различных размерах матриц (квадратных и прямоугольных).

### 7.2 Performance

Тест на данных, состоящих из 100 000 000 символов:

| Mode | Count | Time, s   | Speedup | Efficiency |
|------|-------|-----------|---------|------------|
| seq  | 1     | 0.132     | 1.00    | N/A        |
| mpi  | 2     | 0.196     | 0.67    | 33.5%      |
| mpi  | 4     | 0.174     | 0.76    | 19.0%      |

## 8. Conclusions
К сожалению, реализация mpi версии не дала ожидаемых результатов. Последовательная версия оказалась более быстрой. Причиной этого могут быть накладные расходы на копирование данных.

## 9. References
1. Лекции, практики курса "Параллельное программирование"

## Appendix (Optional)
```cpp
bool MaslovaURowMatrVecMultMPI::RunImpl() {
  int rank;
  int proc_size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &proc_size);

  uint64_t rows = 0;
  uint64_t cols = 0;
  if (rank == 0) {
    rows = static_cast<uint64_t>(GetInput().first.rows);
    cols = static_cast<uint64_t>(GetInput().first.cols);
  }
  MPI_Bcast(&rows, 1, MPI_UINT64_T, 0, MPI_COMM_WORLD);
  MPI_Bcast(&cols, 1, MPI_UINT64_T, 0, MPI_COMM_WORLD); //рассылаем размеры матрицы

  MPI_Datatype row_type; //для удобства создадим новый тип строка
  MPI_Type_contiguous(static_cast<int>(cols), MPI_DOUBLE, &row_type);
  MPI_Type_commit(&row_type);

  std::vector<double> vec(cols);
  if (rank == 0) vec = GetInput().second;
  MPI_Bcast(vec.data(), static_cast<int>(cols), MPI_DOUBLE, 0, MPI_COMM_WORLD); //рассылаем вектор

  std::vector<int> row_cnt(proc_size); //сколько срок получит каждый процесс
  std::vector<int> row_start(proc_size); //с какой строки начинается порция процесса

  int q = static_cast<int>(rows) / proc_size;
  int r = static_cast<int>(rows) % proc_size;

  for (int i = 0; i < proc_size; ++i) {
    row_cnt[i] = q + (i < r ? 1 : 0);
    row_start[i] = (i == 0) ? 0 : row_start[i - 1] + row_cnt[i - 1];
  }

  int local_rows = row_cnt[rank];
  std::vector<double> local_matrix(local_rows * cols);
  
  MPI_Scatterv(rank == 0 ? GetInput().first.data.data() : nullptr, 
               row_cnt.data(), row_start.data(), row_type, 
               local_matrix.data(), static_cast<int>(local_rows * cols), MPI_DOUBLE, 
               0, MPI_COMM_WORLD); //рассылаем строки матрицы

  std::vector<double> local_res(local_rows, 0.0);
  for (int i = 0; i < local_rows; ++i) {
    for (size_t j = 0; j < cols; ++j) {
      local_res[i] += local_matrix[i * cols + j] * vec[j]; //вычисляем локально
    }
  }

  if (rank == 0) GetOutput().resize(rows);
  
  MPI_Gatherv(local_res.data(), local_rows, MPI_DOUBLE,
              rank == 0 ? GetOutput().data() : nullptr, 
              row_cnt.data(), row_start.data(), MPI_DOUBLE, 
              0, MPI_COMM_WORLD); //собираем результаты

  MPI_Type_free(&row_type); //очищаем созданный ип

  return true;
}
```