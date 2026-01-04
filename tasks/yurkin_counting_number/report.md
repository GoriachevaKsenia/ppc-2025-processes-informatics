# Отчёт

# Подсчёт количества букв во входных данных. Последовательная и MPI реализации

- **Студент:** Юркин Георгий Алексеевич, группа 3823Б1ФИ1  
- **Технология:** SEQ–MPI  
- **Вариант:** 22  

---

## 1. Введение

В данной работе рассматривается задача подсчёта количества букв (A–Z, a–z) во входных данных. Задача является типовой для анализа массивов данных и хорошо подходит для демонстрации возможностей последовательных и параллельных вычислений.

Целью работы является реализация последовательной (SEQ) и параллельной (MPI) версий алгоритма подсчёта, проверка корректности их работы и сравнение производительности.
---

## 2. Постановка задачи

Необходимо выполнить подсчёт количества букв во входных данных.

### Входные данные
- `input`— массив символов (std::vector<char>) 

### Выходные данные
- `result` — количество букв в массиве  

### Требования
- Реализация SEQ и MPI версий  
- Корректная работа при любом числе процессов  
- Совпадение результатов SEQ и MPI реализаций  

---

## 3. Последовательный алгоритм

Последовательная версия алгоритма выполняет вычисления в одном потоке:

1. Получение входных данных  
2. Инициализация счётчика  
3. Последовательный перебор входных данных  
4. Увеличение счётчика при выполнении условия  
5. Возврат итогового значения  

Алгоритм имеет линейную вычислительную сложность и не требует дополнительной памяти.

---

## 4. Параллельный алгоритм (MPI)

### Идея распараллеливания

- Входные данные логически делятся между MPI-процессами  
- Для распределения данных используется MPI_Scatterv, что позволяет каждому процессу получить только свой участок массива; корневой процесс предварительно вычисляет массивы sendcounts и displs  
- Каждый процесс подсчитывает количество букв в своей части
- Глобальный результат формируется с помощью `MPI_Allreduce`  

### Синхронизация
- Используется `MPI_COMM_WORLD`  
- Объединение частичных результатов выполняется коллективной операцией MPI_Allreduce

---
### Псевдокод

```
MPI_Comm_rank(MPI_COMM_WORLD, &rank)
MPI_Comm_size(MPI_COMM_WORLD, &size)

if rank == 0:
    N = GetInput().size()
MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD)

compute sendcounts[] and displs[] based on N and size

local_buf = vector<char>(sendcounts[rank])

MPI_Scatterv(input_buffer, sendcounts, displs, MPI_CHAR,
             local_buf, sendcounts[rank], MPI_CHAR,
             0, MPI_COMM_WORLD)

local_count = 0
for c in local_buf:
    if isalpha(static_cast<unsigned char>(c)):
        local_count += 1

global_count = 0
MPI_Allreduce(&local_count, &global_count, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD)

GetOutput() = global_count




```
---

## 5. Детали реализации

### Структура проекта
- `ops_seq.cpp` — последовательная реализация  
- `ops_mpi.cpp` — параллельная MPI-реализация  
- `common.hpp` — описание типов входных и выходных данных  
- `tests/functional` — функциональные тесты  
- `tests/performance` — тесты производительности  

### Особенности
- MPI-реализация использует Scatterv для распределения входного массива между процессами
- После локального подсчёта букв используется `MPI_Allreduce` для получения глобального результата
- Логика SEQ и MPI версий максимально совпадает  

---

## 6. Экспериментальная установка

### Аппаратное обеспечение
- CPU: AMD RYZEN 5500U
- RAM: 16 ГБ  
- OS: Windows 10   
- MPI: OpenMPI / MS-MPI  

### Параметры тестирования
- Различные размеры входных данных  
- Количество процессов: 1, 2, 4  

---

## 7. Результаты

### 7.1 Корректность

- Результаты SEQ и MPI совпадают  
- Функциональные тесты пройдены  
- Ошибки синхронизации отсутствуют  

### 7.2 Производительность

Для оценки производительности были проведены замеры времени выполнения последовательной и MPI-версий алгоритма подсчёта. Тестирование выполнялось на больших объёмах входных данных, сгенерированных автоматически в тестах производительности.

| Mode | Processes | Размер входных данных | Time (s) | Speedup |
|------|-----------|-----------------------|----------|---------|
| seq  | 1         | 1 × 10⁶               | 0.095    | 1.00    |
| mpi  | 1         | 1 × 10⁶               | 0.102    | 0.93    |
| mpi  | 2         | 1 × 10⁶               | 0.056    | 1.70    |
| mpi  | 4         | 1 × 10⁶               | 0.031    | 3.06    |

**Обсуждение:**  
MPI-версия показывает ускорение при использовании нескольких процессов за счёт распараллеливания подсчёта. При одном процессе MPI уступает последовательной версии из-за накладных расходов на инициализацию и коммуникации. С ростом числа процессов и размера входных данных эффективность параллелизации возрастает, однако масштабируемость ограничена затратами на коллективные операции (`MPI_Allreduce`).


## 8. Заключение

В ходе работы была реализована задача подсчёта количества элементов в последовательной и параллельной (MPI) версиях.

- Подтверждена корректность работы  
- Показана применимость MPI для задач подсчёта  
- Получено ускорение на больших входных данных  

---

## 9. References

1. MPI Standard — https://www.mpi-forum.org  
2. cppreference.com — https://en.cppreference.com  
3. Документация OpenMPI / MS-MPI  


## Appendix (Optional)

### MPI-код

```
int world_rank = 0, world_size = 0;
MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
MPI_Comm_size(MPI_COMM_WORLD, &world_size);

int total_size = 0;
if (world_rank == 0) {
    total_size = static_cast<int>(GetInput().size());
}
MPI_Bcast(&total_size, 1, MPI_INT, 0, MPI_COMM_WORLD);

std::vector<int> sendcounts;
std::vector<int> displs;

if (world_rank == 0) {
    sendcounts.assign(world_size, 0);
    displs.assign(world_size, 0);

    std::size_t n = static_cast<std::size_t>(total_size);
    std::size_t chunk = n / static_cast<std::size_t>(world_size);
    std::size_t rem = n % static_cast<std::size_t>(world_size);

    std::size_t offset = 0;
    for (int r = 0; r < world_size; ++r) {
        std::size_t add = chunk + (static_cast<std::size_t>(r) < rem ? 1U : 0U);
        sendcounts[r] = static_cast<int>(add);
        displs[r] = static_cast<int>(offset);
        offset += add;
    }
}

int recvcount = 0;
if (world_rank == 0) {
    MPI_Scatter(sendcounts.data(), 1, MPI_INT,
                &recvcount, 1, MPI_INT, 0, MPI_COMM_WORLD);
} else {
    MPI_Scatter(nullptr, 1, MPI_INT,
                &recvcount, 1, MPI_INT, 0, MPI_COMM_WORLD);
}

if (recvcount < 0) recvcount = 0;

std::vector<char> local_buf(static_cast<std::size_t>(recvcount));

if (world_rank == 0) {
    MPI_Scatterv(GetInput().data(),
                 sendcounts.data(),
                 displs.data(),
                 MPI_CHAR,
                 local_buf.data(),
                 recvcount,
                 MPI_CHAR,
                 0,
                 MPI_COMM_WORLD);
} else {
    MPI_Scatterv(nullptr,
                 nullptr,
                 nullptr,
                 MPI_CHAR,
                 local_buf.data(),
                 recvcount,
                 MPI_CHAR,
                 0,
                 MPI_COMM_WORLD);
}

int local_count = 0;
for (char c : local_buf) {
    if (std::isalpha(static_cast<unsigned char>(c)) != 0) {
        ++local_count;
    }
}

int global_count = 0;
MPI_Allreduce(&local_count, &global_count, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

GetOutput() = global_count;



```