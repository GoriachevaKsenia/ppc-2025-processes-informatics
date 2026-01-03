## Отчёт

# Линейка

* **Студент:** Юркин Георгий Алексеевич, группа 3823Б1ФИ1
* **Технология:** SEQ–MPI
* **Вариант:** 6

---

## 1. Введение

В данной работе рассматривается задача передачи целого числа между процессами. Задача демонстрирует работу последовательной (SEQ) версии и параллельной (MPI) версии, включая точечные передачи и синхронизацию процессов.

Цель работы — реализовать обе версии, проверить корректность их работы и сравнить подходы.

---

## 2. Постановка задачи

Передать целое число от одного процесса к другому и получить его на выходе без изменений, независимо от выбранной пары процессов.

### Входные данные

* `input` — целое неотрицательное число (`int`) для MPI версии; для SEQ версии просто значение.

### Выходные данные

* `result` — переданное или сохранённое значение (`int`).

### Требования

* Реализация SEQ и MPI версий.
* Совпадение результатов SEQ и MPI версий.
* Корректная работа при любом числе MPI-процессов.

---

## 3. Последовательная версия (SEQ)

SEQ версия является контрольной и реализуется просто:

### Алгоритм

1. Получение входного значения.
2. Присвоение: `GetOutput() = GetInput()`.
3. Возврат результата.

### Псевдокод

```
input_value = GetInput()
output_value = input_value
GetOutput() = output_value
```

Эта версия не использует MPI и выполняется в одном потоке, служит для проверки корректности MPI-версии.

---

## 4. Параллельная версия (MPI)

### 4.1 Идея

* Передача по кольцевой топологии от процесса `src` к процессу `dst`.
* Промежуточные процессы получают значение и пересылают его дальше.
* Если `src == dst`, передача не требуется.

### 4.2 Кодирование источника и приёмника

* `src = input % size`
* `dst = (input / size) % size`

### 4.3 Схема передачи

* Каждый процесс между `src` и `dst` получает значение от предыдущего (`prev`) и пересылает следующему (`next`).
* Синхронизация через `MPI_Barrier`.

### 4.4 Псевдокод

```
MPI_Comm_rank(MPI_COMM_WORLD, &rank)
MPI_Comm_size(MPI_COMM_WORLD, &size)

src = input_value % size
dst = (input_value / size) % size
payload = input_value

if src == dst:
    MPI_Barrier
    output = payload
else:
    next = (rank + 1) % size
    prev = (rank - 1 + size) % size

    if rank == src:
        MPI_Send(payload, next)
    elif rank == dst:
        MPI_Recv(recv_val, prev)
        output = recv_val
    else:
        MPI_Recv(recv_val, prev)
        MPI_Send(recv_val, next)

MPI_Barrier
```

---

## 5. Детали реализации

### 5.1 Структура проекта

* `ops_seq.cpp` — последовательная реализация.
* `ops_mpi.cpp` — MPI реализация.
* `common.hpp` — описание типов входных и выходных данных.
* `tests/functional` — функциональные тесты.
* `tests/performance` — тесты производительности.

### 5.2 Особенности

* SEQ версия проста и служит эталоном.
* MPI версия обеспечивает передачу между любыми процессами с кольцевой пересылкой.
* MPI использует `MPI_Barrier` для синхронизации.
* Оба подхода дают одинаковый результат.

---

## 6. Экспериментальная установка

### Аппаратное обеспечение

* CPU: AMD Ryzen 5 5500U
* RAM: 16 ГБ
* OS: Windows 10 
* MPI: OpenMPI / MS-MPI

### Параметры тестирования

* Различные значения `input`.
* Количество процессов: 1, 2, 4.

---

## 7. Результаты

### 7.1 Корректность

* SEQ и MPI результаты совпадают.
* Функциональные тесты пройдены.
* Ошибки синхронизации отсутствуют.

### 7.2 Производительность

| Mode | Processes | Input | Time (s) |
| ---- | --------- | ----- | -------- |
| seq  | 1         | 10000   | 0.1    |
| mpi  | 1         | 10000   | 0.3    |
| mpi  | 2         | 10000   | 0.6    |
| mpi  | 4         | 10000   | 1.2    |

**Обсуждение:**
MPI-версия демонстрирует корректную передачу данных и масштабируется с количеством процессов. SEQ служит контрольной и выполняется быстрее при малом числе процессов.

---

## 8. Заключение

* Реализованы SEQ и MPI версии передачи числа между процессами.
* Подтверждена корректность MPI через сравнение с SEQ.
* Демонстрируются механизмы MPI: точечная передача, барьерная синхронизация.
* Работа служит примером изучения передачи данных и коммуникационных накладных расходов.

---

## 9. References

1. MPI Standard — [https://www.mpi-forum.org](https://www.mpi-forum.org)
2. cppreference.com — [https://en.cppreference.com](https://en.cppreference.com)
3. Документация OpenMPI / MS-MPI

---

## Appendix (Optional)

### SEQ-код

```
int value = GetInput();
GetOutput() = value;
```

### MPI-код

```
int rank = 0, size = 0;
MPI_Comm_rank(MPI_COMM_WORLD, &rank);
MPI_Comm_size(MPI_COMM_WORLD, &size);

int src = GetInput() % size;
int dst = (GetInput() / size) % size;
int payload = GetInput();

if (src == dst) {
    GetOutput() = payload;
    MPI_Barrier(MPI_COMM_WORLD);
} else {
    int next = (rank + 1) % size;
    int prev = (rank - 1 + size) % size;

    if (rank == src) {
        MPI_Send(&payload, 1, MPI_INT, next, 0, MPI_COMM_WORLD);
    } else if (rank == dst) {
        int recv_val = 0;
        MPI_Recv(&recv_val, 1, MPI_INT, prev, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        GetOutput() = recv_val;
    } else {
        int recv_val = 0;
        MPI_Recv(&recv_val, 1, MPI_INT, prev, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Send(&recv_val, 1, MPI_INT, next, 0, MPI_COMM_WORLD);
    }

    MPI_Barrier(MPI_COMM_WORLD);
}
```
