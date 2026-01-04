# Отчёт по лабораторной работе  
## «Параллельное нахождение максимальных значений по столбцам матрицы с использованием MPI»

**Дисциплина:** Параллельное программирование  
**Преподаватель:** Нестеров Александр Юрьевич и Оболенский Арсений Андреевич  
**Студент:** Косолапов Виталий Андреевич 3823Б1ФИ1
**Вариант:** 9
---

## Введение

Вычисление кратных интегралов является одной из важнейших задач вычислительной математики, которая находит применение в физике, статистике, машинном обучении и других научных областях. При увеличении размерности и точности вычислений последовательные методы становятся недостаточно эффективными из-за значительных временных затрат. Параллельные вычисления позволяют распределить нагрузку между несколькими вычислительными узлами, что существенно сокращает время решения задачи.

В данной работе реализованы последовательная и параллельная версии вычисления двойных интегралов методом Симпсона с использованием технологии MPI (Message Passing Interface). Цель работы — освоение принципов параллельного программирования, сравнение производительности и проверка корректности параллельной реализации.


---

## Постановка задачи

Необходимо вычислить **двойной интеграл** от заданной функции \( f(x, y) \) по прямоугольной области \( [a, b] \times [c, d] \) с использованием **составной формулы Симпсона**.

Дана функция \( f(x, y) \), область интегрирования и чётное число шагов интегрирования \( N \). Требуется вычислить приближённое значение интеграла:

\[
I \approx \int_{a}^{b} \int_{c}^{d} f(x, y) \, dy \, dx
\]

### Требуется:
1. Реализовать **последовательную версию** вычисления.  
2. Реализовать **параллельную версию** с использованием **MPI**.  
3. Провести сравнение времени выполнения и подтвердить корректность вычислений.  

---

## Описание алгоритма

Метод Симпсона для двойного интеграла реализуется следующим образом:

1. Задать область интегрирования \( [a, b] \times [c, d] \) и чётное число шагов \( N \).  
2. Вычислить шаги сетки:
   \[
   h_x = \frac{b - a}{N}, \quad h_y = \frac{d - c}{N}
   \]
3. Для каждого узла сетки \( (x_i, y_j) \) вычислить весовой коэффициент по формуле Симпсона:
   - На границах: 1  
   - Для чётных индексов внутри: 2  
   - Для нечётных индексов внутри: 4  
4. Вычислить приближённое значение интеграла:
   \[
   I \approx \frac{h_x \cdot h_y}{9} \sum_{i=0}^{N} \sum_{j=0}^{N} w_i \cdot w_j \cdot f(x_i, y_j)
   \]

5. Вернуть полученное значение как результат.

---

## Описание схемы параллельного алгоритма

В параллельной версии с использованием **MPI**:

1. **Инициализация MPI** — каждый процесс получает свой ранг `rank` и общее число процессов `processes_count`.  
2. **Разделение данных:**  
   - Строки сетки по оси \( x \) равномерно распределяются между процессами.  
   - Если число строк не делится нацело, лишние строки распределяются по первым процессам.    
3. **Вычисление частичных результатов:**  
   Каждый процесс вычисляет сумму по своей части строк сетки.  
4. **Сбор результатов:**  
   Используется `MPI_Reduce` для суммирования частичных результатов на процессе с рангом 0.
5. **Вывод результата и завершение работы MPI.**

---

## Описание программной реализации (MPI-версия)

Реализация выполнена на языке **C++20** с использованием **библиотеки MPI**.  
Класс `KosolapovVCalcMultIntegralsMSimpsonMPI` реализует интерфейс параллельного вычисления интеграла.
  
Основные этапы:
- Инициализация MPI и валидация входных данных;  
- Распределение диапазонов строк сетки между процессами;  
- Вычисление локальной части интеграла;  
- Сбор результатов с помощью `MPI_Reduce`;  
- Вывод итогового значения.

### Ключевые функции:
- `MPI_Comm_rank`, `MPI_Comm_size` — определяют номер процесса и общее количество процессов;  
- `MPI_Send` — отправляет сообщение(данные) другому процессу;
- `MPI_Recv` — принимает сообщение(данные) от другого процесса;
- `MPI_Reduce` — коллективная операция, которая выполняет редукцию (суммирование, максимум и т.д.) над данными всех процессов и рассылает результат одному процессу;
- `MPI_Bcast` - рассылает данные от одного процесса всем остальным процессам.

### Валидация данных
Проверяется, что число шагов чётное и положительное, а идентификатор функции находится в допустимом диапазоне.

---

## Результаты экспериментов

### Условия экспериментов
- Число шагов: \( N = 10000, 20000 \).  
- Функции: \( f_1(x, y) = x^2 + y^2 \).
- Диапазоны: \($\int_0^1 dx \int_0^1 dy \ f(x,y)$)\
- Среда выполнения: Windows, MPI (4 процесса).  
- Измерение времени проводилось встроенными средствами тестового фреймворка GoogleTest.  

---

### Результаты при числе шагов равном 10000


| Режим выполнения| Кол-во процессов | Время (сек) | Ускорение | Эффективность |
|-----------------|------------------|-------------|-----------|---------------|
| SEQ pipeline    | 1                | 0.72899     | 1.0       | N/A           |
| SEQ task_run    | 1                | 0.71234     | 1.0       | N/A           |
| MPI pipeline    | 4                | 0.21206     | 3.44      | 86%           |
| MPI task_run    | 4                | 0.21024     | 3.36      | 84%           |
| MPI pipeline    | 6                | 0.14897     | 4.89      | 81.55%        |
| MPI task_run    | 6                | 0.14649     | 4.86      | 81.04%        |

---

### Результаты при числе шагов равном 20000


| Режим выполнения| Кол-во процессов | Время (сек) | Ускорение | Эффективность |
|-----------------|------------------|-------------|-----------|---------------|
| SEQ pipeline    | 1                | 2.8462      | 1.0       | N/A           |
| SEQ task_run    | 1                | 2.83309     | 1.0       | N/A           |
| MPI pipeline    | 4                | 0.82486     | 3.43      | 85.87%           |
| MPI task_run    | 4                | 0.80162     | 3.53      | 88.36%        |

---


## Подтверждение корректности
Результаты последовательной и параллельной версий совпадают для всех протестированных вариаций функций и количества шагов. Ошибок не наблюдалось, при тестировании возвращался результат в из допустимой области. Таким образом, реализация корректна.

---

## Выводы
1. Реализованы две версии алгоритма — последовательная и MPI.  
2. Параллельная реализация успешно масштабируется и даёт ускорение при увеличении размера данных.  
3. Параллельная реализация работает с одинаковой эффективностью при разном количестве процессов.  
4. Алгоритм корректен и демонстрирует ожидаемое поведение при всех размерах входных данных.  

---

## Заключение
В ходе лабораторной работы были реализованы последовательный и параллельный алгоритмы вычисления двойных интегралов методом Симпсона с использованием MPI. Проведён сравнительный анализ производительности, подтверждена корректность вычислений. Результаты показывают эффективность параллельных вычислений для ресурсоёмких задач интегрирования.

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

#include "kosolapov_v_calc_mult_integrals_m_simpson/seq/include/ops_seq.hpp"

#include <cmath>
#include <tuple>

#include "kosolapov_v_calc_mult_integrals_m_simpson/common/include/common.hpp"

namespace kosolapov_v_calc_mult_integrals_m_simpson {

KosolapovVCalcMultIntegralsMSimpsonSEQ::KosolapovVCalcMultIntegralsMSimpsonSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = InType(in);
  GetOutput() = 0.0;
}

bool KosolapovVCalcMultIntegralsMSimpsonSEQ::ValidationImpl() {
  int steps = std::get<0>(GetInput());
  int func_id = std::get<1>(GetInput());
  return steps > 0 && (steps % 2 == 0) && func_id >= 1 && func_id <= 4;
}

bool KosolapovVCalcMultIntegralsMSimpsonSEQ::PreProcessingImpl() {
  return true;
}

bool KosolapovVCalcMultIntegralsMSimpsonSEQ::RunImpl() {
  int steps = std::get<0>(GetInput());
  int func_id = std::get<1>(GetInput());
  std::tuple<double, double, double, double> temp = GetBounds(func_id);
  double a = std::get<0>(temp);
  double b = std::get<1>(temp);
  double c = std::get<2>(temp);
  double d = std::get<3>(temp);
  double integral = SimpsonIntegral(func_id, steps, a, b, c, d);
  GetOutput() = integral;
  return true;
}

bool KosolapovVCalcMultIntegralsMSimpsonSEQ::PostProcessingImpl() {
  return true;
}

double KosolapovVCalcMultIntegralsMSimpsonSEQ::Function1(double x, double y) {
  // f(x,y) = x^2 + y^2
  return (x * x) + (y * y);
}
double KosolapovVCalcMultIntegralsMSimpsonSEQ::Function2(double x, double y) {
  // f(x,y) = sin(x) * cos(y)
  return std::sin(x) * std::cos(y);
}
double KosolapovVCalcMultIntegralsMSimpsonSEQ::Function3(double x, double y) {
  // f(x,y) = exp(-(x^2 + y^2))
  return std::exp(-((x * x) + (y * y)));
}
double KosolapovVCalcMultIntegralsMSimpsonSEQ::Function4(double x, double y) {
  // f(x,y) = sin(x + y)
  return std::sin(x + y);
}
double KosolapovVCalcMultIntegralsMSimpsonSEQ::CallFunction(int func_id, double x, double y) {
  switch (func_id) {
    case 1:
      return Function1(x, y);
    case 2:
      return Function2(x, y);
    case 3:
      return Function3(x, y);
    case 4:
      return Function4(x, y);
    default:
      return Function1(x, y);
  }
}
std::tuple<double, double, double, double> KosolapovVCalcMultIntegralsMSimpsonSEQ::GetBounds(int func_id) {
  switch (func_id) {
    case 1:
      return {0.0, 1.0, 0.0, 1.0};
    case 2:
      return {0.0, kPi, 0.0, kPi / 2.0};
    case 3:
      return {-1.0, 1.0, -1.0, 1.0};
    case 4:
      return {0.0, kPi, 0.0, kPi};
    default:
      return {0.0, 1.0, 0.0, 1.0};
  }
}
double KosolapovVCalcMultIntegralsMSimpsonSEQ::SimpsonIntegral(int func_id, int steps, double a, double b, double c,
                                                               double d) {
  double hx = (b - a) / steps;
  double hy = (d - c) / steps;
  double result = 0.0;
  for (int i = 0; i <= steps; i++) {
    double x = a + (i * hx);
    double wx = GetSimpsonWeight(i, steps);
    for (int j = 0; j <= steps; j++) {
      double y = c + (j * hy);
      double wy = GetSimpsonWeight(j, steps);
      result += (wx * wy) * CallFunction(func_id, x, y);
    }
  }
  result *= (hx * hy) / 9.0;
  return result;
}
double KosolapovVCalcMultIntegralsMSimpsonSEQ::GetSimpsonWeight(int index, int steps) {
  if (index == 0 || index == steps) {
    return 1.0;
  }
  return (index % 2 == 0) ? 2.0 : 4.0;
}

}  // namespace kosolapov_v_calc_mult_integrals_m_simpson


```
---
### MPI-версия
```cpp
#include "kosolapov_v_calc_mult_integrals_m_simpson/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <cmath>
#include <tuple>

#include "kosolapov_v_calc_mult_integrals_m_simpson/common/include/common.hpp"

namespace kosolapov_v_calc_mult_integrals_m_simpson {

KosolapovVCalcMultIntegralsMSimpsonMPI::KosolapovVCalcMultIntegralsMSimpsonMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = InType(in);
  GetOutput() = 0.0;
}

bool KosolapovVCalcMultIntegralsMSimpsonMPI::ValidationImpl() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if (rank == 0) {
    int steps = std::get<0>(GetInput());
    int func_id = std::get<1>(GetInput());
    return steps > 0 && (steps % 2 == 0) && func_id >= 1 && func_id <= 4;
  }
  return true;
}

bool KosolapovVCalcMultIntegralsMSimpsonMPI::PreProcessingImpl() {
  return true;
}

bool KosolapovVCalcMultIntegralsMSimpsonMPI::RunImpl() {
  int processes_count = 0;
  int rank = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &processes_count);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  int steps = 0;
  int func_id = 0;
  if (rank == 0) {
    steps = std::get<0>(GetInput());
    func_id = std::get<1>(GetInput());
  }
  MPI_Bcast(&steps, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&func_id, 1, MPI_INT, 0, MPI_COMM_WORLD);
  int rows_per_process = (steps + 1) / processes_count;
  int remainder = (steps + 1) % processes_count;

  int start_i = 0;
  int end_i = 0;
  if (rank < remainder) {
    start_i = rank * (rows_per_process + 1);
    end_i = start_i + rows_per_process;
  } else {
    start_i = (remainder * (rows_per_process + 1)) + ((rank - remainder) * rows_per_process);
    end_i = start_i + rows_per_process - 1;
  }
  if (rank == processes_count - 1) {
    end_i = steps;
  }
  std::tuple<double, double, double, double> temp = GetBounds(func_id);
  double a = std::get<0>(temp);
  double b = std::get<1>(temp);
  double c = std::get<2>(temp);
  double d = std::get<3>(temp);
  double local_sum = ComputePartialSimpsonIntegral(func_id, steps, a, b, c, d, start_i, end_i);
  double global_sum = 0.0;
  MPI_Reduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  double result = 0.0;
  if (rank == 0) {
    double hx = (b - a) / steps;
    double hy = (d - c) / steps;
    result = global_sum * (hx * hy) / 9.0;
  }
  MPI_Bcast(&result, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  GetOutput() = result;
  return true;
}

bool KosolapovVCalcMultIntegralsMSimpsonMPI::PostProcessingImpl() {
  return true;
}

double KosolapovVCalcMultIntegralsMSimpsonMPI::Function1(double x, double y) {
  // f(x,y) = x^2 + y^2
  return (x * x) + (y * y);
}
double KosolapovVCalcMultIntegralsMSimpsonMPI::Function2(double x, double y) {
  // f(x,y) = sin(x) * cos(y)
  return std::sin(x) * std::cos(y);
}
double KosolapovVCalcMultIntegralsMSimpsonMPI::Function3(double x, double y) {
  // f(x,y) = exp(-(x^2 + y^2))
  return std::exp(-((x * x) + (y * y)));
}
double KosolapovVCalcMultIntegralsMSimpsonMPI::Function4(double x, double y) {
  // f(x,y) = sin(x + y)
  return std::sin(x + y);
}
double KosolapovVCalcMultIntegralsMSimpsonMPI::CallFunction(int func_id, double x, double y) {
  switch (func_id) {
    case 1:
      return Function1(x, y);
    case 2:
      return Function2(x, y);
    case 3:
      return Function3(x, y);
    case 4:
      return Function4(x, y);
    default:
      return Function1(x, y);
  }
}
std::tuple<double, double, double, double> KosolapovVCalcMultIntegralsMSimpsonMPI::GetBounds(int func_id) {
  switch (func_id) {
    case 1:
      return {0.0, 1.0, 0.0, 1.0};
    case 2:
      return {0.0, kPi, 0.0, kPi / 2.0};
    case 3:
      return {-1.0, 1.0, -1.0, 1.0};
    case 4:
      return {0.0, kPi, 0.0, kPi};
    default:
      return {0.0, 1.0, 0.0, 1.0};
  }
}
double KosolapovVCalcMultIntegralsMSimpsonMPI::ComputePartialSimpsonIntegral(int func_id, int steps, double a, double b,
                                                                             double c, double d, int start_i,
                                                                             int end_i) {
  double hx = (b - a) / steps;
  double hy = (d - c) / steps;
  double local_sum = 0.0;
  for (int i = start_i; i <= end_i; i++) {
    double x = a + (i * hx);
    double wx = GetSimpsonWeight(i, steps);
    for (int j = 0; j <= steps; j++) {
      double y = c + (j * hy);
      double wy = GetSimpsonWeight(j, steps);
      local_sum += (wx * wy) * CallFunction(func_id, x, y);
    }
  }
  return local_sum;
}
double KosolapovVCalcMultIntegralsMSimpsonMPI::GetSimpsonWeight(int index, int steps) {
  if (index == 0 || index == steps) {
    return 1.0;
  }
  return (index % 2 == 0) ? 2.0 : 4.0;
}

}  // namespace kosolapov_v_calc_mult_integrals_m_simpson


```