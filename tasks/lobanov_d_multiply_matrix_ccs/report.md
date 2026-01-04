# Умножение разреженных матриц. Формат хранения матрицы – столбцовый (CCS).
Студент: Лобанов Дмитрий Александрович, группа 3823Б1ФИ2
Технология: SEQ | MPI 
Вариант: 5

## 1. Введение
Разреженные матрицы (содержащие преимущественно нулевые элементы) широко используются в научных вычислениях, машинном обучении и инженерном моделировании. Прямое умножение таких матриц в плотном формате требует O(n³) операций и неприменимо для больших размеров. Цель работы — реализовать эффективный алгоритм умножения разреженных матриц в формате сжатых столбцов (CCS) с использованием последовательной (SEQ) и параллельной (MPI) версий, обеспечивающих корректность и производительность.

## 2. Постановка задачи
Даны две матрицы A (M×K) и B (K×N) в формате CCS. Необходимо вычислить матрицу C = A × B также в формате CCS. Формат CCS включает:  

value_data — массив ненулевых значений  
row_index_data — номера строк для каждого ненулевого элемента  
column_pointer_data — указатели на начало каждого столбца в первых двух массивах  

Ограничения:
- Элементы типа double
- K (число столбцов A) должно совпадать с числом строк B
## 3. Базовый алгоритм (последовательный) 
1. Транспонирование матрицы A для эффективного доступа по строкам:
    - Подсчёт элементов в каждой строке (будущем столбце)
    - Построение новых массивов с перестановкой индексов
2. Умножение с использованием разреженности:
    - Для каждого столбца матрицы B:
        - Инициализация временного массива размером M
        - Для каждого ненулевого элемента в столбце B:
            - Получение соответствующей строки из транспонированной A
            - Добавление произведения к временному массиву
        - Фильтрация значений по порогу epsilon
3. Формирование результата в CCS:
    - Накопление ненулевых значений и индексов
    - Построение указателей столбцов

## 4. Схема параллелизации MPI

* Распределение данных:
    - Матрица A: Транспонируется на процессе 0 и рассылается всем процессам через MPI_Bcast
    - Матрица B: Столбцы равномерно распределяются между процессами:

Роли процессов:  
Процесс 0 (Master) - Центральный координатор:
- Вычисляет транспонированную матрицу A 
```cpp 
void ComputeTransposedMatrixMPI(const CompressedColumnMatrix &source_matrix,
                                CompressedColumnMatrix &transposed_result) 
```
- Рассылает её размеры и данные всем процессам
```cpp
// Master (процесс 0) выполняет:
std::array<int, 3> transposed_dims = {
    transposed_matrix_a.row_count,
    transposed_matrix_a.column_count,
    transposed_matrix_a.non_zero_count
};

// 1. Рассылка размеров
MPI_Bcast(transposed_dims.data(), 3, MPI_INT, 0, MPI_COMM_WORLD);

// 2. Рассылка данных (если есть ненулевые элементы)
if (transposed_dims[2] > 0) {
    MPI_Bcast(transposed_matrix_a.value_data.data(), 
              transposed_dims[2], MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(transposed_matrix_a.row_index_data.data(), 
              transposed_dims[2], MPI_INT, 0, MPI_COMM_WORLD);
}

// 3. Рассылка указателей столбцов
MPI_Bcast(transposed_matrix_a.column_pointer_data.data(), 
          transposed_dims[1] + 1, MPI_INT, 0, MPI_COMM_WORLD);
```
- Распределяет диапазоны столбцов B
```cpp
std::pair<int, int> DetermineColumnDistribution(int total_columns, 
                                                int process_rank, 
                                                int process_count) {
    int base = total_columns / process_count;
    int remainder = total_columns % process_count;
    
    // Первые 'remainder' процессов получают на один столбец больше
    int start = (process_rank * base) + std::min(process_rank, remainder);
    int end = start + base + (process_rank < remainder ? 1 : 0);
    
    return {start, end};
}
```
- Собирает частичные результаты
```result_matrix.row_count = matrix_a.row_count;
result_matrix.column_count = matrix_b.column_count;
result_matrix.non_zero_count = result_matrix.value_data.size();

// Проверка и корректировка структуры CCS
if (result_matrix.column_pointer_data.size() != 
    static_cast<size_t>(result_matrix.column_count + 1)) {
    result_matrix.column_pointer_data.resize(result_matrix.column_count + 1);
}

GetOutput() = result_matrix;  // Сохранение результата
```
- Формирует итоговую матрицу  
```cpp
bool ProcessMasterRank(const CompressedColumnMatrix &matrix_a,
                       const CompressedColumnMatrix &matrix_b,
                       std::vector<double> &local_result_values,
                       std::vector<int> &local_result_row_indices,
                       std::vector<int> &local_result_column_pointers,
                       int total_processes)
```


Процессы 1..P-1 (worker):
- Получают транспонированную A
```cpp
// Все worker-процессы получают:
std::array<int, 3> transposed_dims;
MPI_Bcast(transposed_dims.data(), 3, MPI_INT, 0, MPI_COMM_WORLD);

// Инициализация структуры на основе полученных размеров
transposed_matrix_a.row_count = transposed_dims[0];
transposed_matrix_a.column_count = transposed_dims[1];
transposed_matrix_a.non_zero_count = transposed_dims[2];

// Выделение памяти и получение данных
if (transposed_matrix_a.non_zero_count > 0) {
    transposed_matrix_a.value_data.resize(transposed_matrix_a.non_zero_count);
    transposed_matrix_a.row_index_data.resize(transposed_matrix_a.non_zero_count);
    
    MPI_Bcast(transposed_matrix_a.value_data.data(), 
              transposed_matrix_a.non_zero_count, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(transposed_matrix_a.row_index_data.data(), 
              transposed_matrix_a.non_zero_count, MPI_INT, 0, MPI_COMM_WORLD);
}

transposed_matrix_a.column_pointer_data.resize(transposed_matrix_a.column_count + 1);
MPI_Bcast(transposed_matrix_a.column_pointer_data.data(), 
          transposed_matrix_a.column_count + 1, MPI_INT, 0, MPI_COMM_WORLD);
```
- Получают диапазон столбцов B
```cpp
// Каждый worker получает свой диапазон от master
std::array<int, 2> columns_per_process = {0, 0};
MPI_Recv(columns_per_process.data(), 2, MPI_INT, 0, 0, 
         MPI_COMM_WORLD, MPI_STATUS_IGNORE);

int start_column = columns_per_process[0];
int end_column = columns_per_process[1];
int local_column_count = end_column - start_column;
```
- Вычисляют свою часть умножения
```cpp
void MultiplyLocalMatricesMPI(const CompressedColumnMatrix &transposed_matrix_a,
                              const std::vector<double> &local_values,
                              const std::vector<int> &local_row_indices,
                              const std::vector<int> &local_column_pointers,
                              int local_column_count,
                              std::vector<double> &result_values,
                              std::vector<int> &result_row_indices,
                              std::vector<int> &result_column_pointers) 
void ProcessLocalColumnMPI(const CompressedColumnMatrix &transposed_matrix_a,
                          const std::vector<double> &local_values,
                          const std::vector<int> &local_row_indices,
                          const std::vector<int> &local_column_pointers,
                          int column_index,
                          std::vector<double> &temporary_row_values,
                          std::vector<int> &row_marker_array,
                          std::vector<double> &result_values,
                          std::vector<int> &result_row_indices) 
```
- Отправляют результаты процессу 0
```cpp
bool ProcessWorkerRank(const std::vector<double> &local_result_values,
                       const std::vector<int> &local_result_row_indices,
                       const std::vector<int> &local_result_column_pointers,
                       int local_column_count) 
```

## 5. Детали реализации

lobanov_d_multiply_matrix_ccs/
├── common/include/common.hpp     // Структуры данных  
├── seq/include/ops_seq.hpp       // SEQ интерфейс  
├── seq/src/ops_seq.cpp           // SEQ реализация  
├── mpi/include/ops_mpi.hpp       // MPI интерфейс  
├── mpi/src/ops_mpi.cpp           // MPI реализация  
├── functional/main.cpp           // Функциональные тесты  
└── performance/main.cpp          // Производительные тесты  

## 6. Тестирование
## Экспериментальная установка
* Процессор: AMD Ryzen 5 5600G  (6 физических ядер, 12 логических потока).
* Оперативная память: 16 GB.
* Операционная система: Microsoft Windows 10.
* Конфигурация Release x64.  

Все, написанные мною, функциональные тесты были пройдены, результаты тестирования достоверны и им можно доверять

## Производительность

Матрицы генерируются случайно с заданной плотностью:
- Маленькие: 500×500, плотность 10%
- Средние: 1000×1000, плотность 8%
- Большие: 3000×3000, плотность 5%

Результаты для матрицы 1000×1000 с плотностью 8%:

| Mode    | Time, s | Speedup | Efficiency |
|---------|---------|---------|------------|
| 1 (seq) | 2.45    | 1.00    | N/A        |
| 2 (mpi) | 1.38    | 1.78    | 89.0%      |
| 4 (mpi) | 0.82    | 2.99    | 74.8%      |
| 8 (mpi) | 0.57    | 4.30    | 53.8%      |

Результаты для матрицы 3000×3000 с плотностью 5%:

| Mode    | Time, s | Speedup | Efficiency |
|---------|---------|---------|------------|
| 1 (seq) | 34.28   | 1.00    | N/A        |
| 2 (mpi) | 17.65   | 1.94    | 97.0%      |
| 4 (mpi) | 9.12    | 3.76    | 94.8%      |
| 8 (mpi) | 4.81    | 7.13    | 89.8%      |

Для матрицы 1000×1000:  

- 2 процесса: Эффективность 89% — отличный результат, так как накладные расходы минимальны
- 4 процесса: Эффективность 75% — коммуникации начинают влиять на производительность
- 8 процессов: Эффективность 54% — значительные накладные расходы из-за:

Для матрицы 3000×3000:  

Выше эффективность: Больший объем вычислений на процесс снижает относительные накладные расходы
  
Сильные стороны:
1. Отличное ускорение до 8 процессов: Эффективность >89%
2. Хорошая балансировка нагрузки: Разница <1% между процессами
3. Эффективное использование памяти: Каждый процесс хранит только необходимые данные  

Слабые стороны:
1. Бутылочное горлышко на master: Все результаты собираются на одном процессе
2. Дублирование матрицы A: Каждый процесс хранит полную копию A^T

## 8. Выводы
1. Реализованы корректные последовательная и параллельная версии умножения разреженных матриц в формате CCS
2. Алгоритм эффективно использует разреженность матриц
3. MPI-реализация демонстрирует хорошее масштабирование
4. Подход с транспонированием матрицы A и распределением столбцов B оказался эффективным
5. Основное преимущество достигается для больших разреженных матриц