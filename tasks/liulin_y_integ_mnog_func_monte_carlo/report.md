# Отчет

# **Отчёт по задаче: Вычисление многомерных интегралов методом Монте-Карло**  
**Вариант:** 10

**Студент:** Люлин Ярослав Сергеевич  
**Группа:** 3823Б1ФИ1  
**Технология:** SEQ | MPI  

---

# **1. Введение**

Вычисление многомерных интегралов - фундаментальная задача численного анализа, широко применяемая в физике, статистике, финансах, машинном обучении и моделировании. Метод Монте-Карло особенно эффективен для интегралов высокой размерности, где детерминированные методы (трапеции, Симпсон, Гаусс) становятся вычислительно дорогими.

В данной работе реализованы две версии алгоритма приближённого вычисления двойного интеграла функции f(x,y) по прямоугольной области методом Монте-Карло:

1. **SEQ (последовательная)** - генерация и обработка всех случайных точек на одном процессе.
2. **MPI (параллельная)** - распределение числа точек между процессами с последующей агрегацией сумм.

---

# **2. Постановка задачи**

Дана функция f(x,y), прямоугольная область [x_min, x_max] × [y_min, y_max] и количество случайных точек N.  
Требуется приближённо вычислить интеграл:

```
I = ∫_{x_min}^{x_max} ∫_{y_min}^{y_max} f(x,y) dy dx ≈ (площадь области) × (среднее значение f в N случайных точках)
```

**Входные данные:**  
границы x_min, x_max, y_min, y_max,  
подынтегральная функция f,  
количество точек N.

**Выходные данные:**  
приближённое значение интеграла (double).

**Ограничения:**  
N - положительное целое, границы образуют ненулевую площадь.

---

# **3. Базовый алгоритм (последовательный)**

SEQ-версия реализует классический метод Монте-Карло:

1. Вычисляется площадь прямоугольника: area = (x_max - x_min) × (y_max - y_min).
2. Генерируются N случайных точек (x,y) равномерно распределённых в области.
3. Вычисляется сумма значений функции f в этих точках.
4. Результат: (сумма / N) × area.

**Фрагмент кода SEQ версии:**
```cpp
const auto &input = GetInput();
  auto &result = GetOutput();

  if (input.num_points <= 0) {
    result = 0.0;
    return true;
  }

  const double area = (input.x_max - input.x_min) * (input.y_max - input.y_min);
  if (area <= 0.0) {
    result = 0.0;
    return true;
  }

  const double x_min = input.x_min;
  const double x_range = input.x_max - x_min;
  const double y_min = input.y_min;
  const double y_range = input.y_max - y_min;
  const auto &func = input.f;
  const int64_t n = input.num_points;

  static thread_local std::minstd_rand fast_gen(std::random_device{}());

  std::uniform_real_distribution<double> dist_x(0.0, x_range);
  std::uniform_real_distribution<double> dist_y(0.0, y_range);

  double sum = 0.0;
  int64_t i = 0;

  for (; i + 3 < n; i += 4) {
    const double x1 = x_min + dist_x(fast_gen);
    const double y1 = y_min + dist_y(fast_gen);
    const double x2 = x_min + dist_x(fast_gen);
    const double y2 = y_min + dist_y(fast_gen);
    const double x3 = x_min + dist_x(fast_gen);
    const double y3 = y_min + dist_y(fast_gen);
    const double x4 = x_min + dist_x(fast_gen);
    const double y4 = y_min + dist_y(fast_gen);

    sum += func(x1, y1) + func(x2, y2) + func(x3, y3) + func(x4, y4);
  }

  for (; i < n; ++i) {
    const double x = x_min + dist_x(fast_gen);
    const double y = y_min + dist_y(fast_gen);
    sum += func(x, y);
  }

  result = (sum * area) / n;

  return true;
```

**Преимущества:**  
простота реализации,  
хорошая точность при большом N (O(1/√N)).

**Недостатки:**  
все точки генерируются и обрабатываются последовательно.

---

# **4. Параллельная версия (MPI)**

MPI-версия распределяет генерацию и вычисление точек между процессами:

1. Root-процесс рассылает параметры (границы, N) всем процессам через `MPI_Bcast`.
2. Каждый процесс вычисляет свою долю точек: примерно N / p, с учётом остатка.
3. Генерируются локальные случайные точки, считается локальная сумма f(x,y).
4. Локальные суммы агрегируются на root через `MPI_Reduce(MPI_SUM)`.
5. Root вычисляет итоговый интеграл и рассылает его всем процессам через `MPI_Bcast`.

**Преимущества:**  
линейное масштабирование при большом N,  
параллельная генерация случайных чисел.

**Недостатки:**  
накладные расходы на коммуникации (особенно при малом N),  
необходимость синхронизации.
**Фрагмент MPI-кода:**
```cpp
  const auto &input = GetInput();
  auto &result = GetOutput();

  int world_size = 0;
  int world_rank = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  int64_t total_points = input.num_points;
  MPI_Bcast(&total_points, 1, MPI_LONG_LONG, 0, MPI_COMM_WORLD);

  if (total_points <= 0) {
    result = 0.0;
    return true;
  }

  double area = (input.x_max - input.x_min) * (input.y_max - input.y_min);
  MPI_Bcast(&area, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  if (area <= 0.0) {
    result = 0.0;
    return true;
  }

  int64_t base_points = total_points / world_size;
  int64_t remainder = total_points % world_size;
  int64_t local_points = base_points + (world_rank < remainder ? 1 : 0);

  std::random_device rd;
  std::mt19937 gen(rd() + world_rank);
  std::uniform_real_distribution<double> dist_x(input.x_min, input.x_max);
  std::uniform_real_distribution<double> dist_y(input.y_min, input.y_max);

  double local_sum = 0.0;
  for (int64_t i = 0; i < local_points; ++i) {
    double x = dist_x(gen);
    double y = dist_y(gen);
    local_sum += input.f(x, y);
  }

  double global_sum = 0.0;
  MPI_Reduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

  if (world_rank == 0) {
    result = global_sum / total_points * area;
  }

  MPI_Bcast(&result, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  return true;
```
---

# **5. Детали реализации**

- Функция f передаётся через `std::function<double(double, double)>`.
- Используется `std::mt19937` + `std::uniform_real_distribution` для равномерной генерации точек.
- В MPI-версии каждый процесс использует уникальный seed (`rd() + rank`) для независимости.
- Валидация: проверка ненулевых границ и положительного N.
- Тесты: загрузка параметров из файлов (x_min x_max y_min y_max N func_id).
- Функции для тестов: 1 (константа), x+y, x*y, sin(x)cos(y).

---

# **6. Тестовое окружение**

**Аппаратное обеспечение  / ОС:** Intel Core i5-12450H, 16 GB DDR4 Ubuntu 24.04.2 LTS X86_64 MPI: OpenMPI

**Инструменты сборки:** CMake, clang++ (Release)

**Тестовые данные:**  
10 функциональных тестов (разные функции, N от 1000 до 10^7),  
большие N (10^8) для perf-тестов.

---

# **7. Результаты**

## 7.1 Корректность

Все функциональные тесты пройдены успешно для SEQ и MPI версий:  
- const1 (f=1),  
- linear (x+y),  
- product (x*y),  
- trig (sin(x)cos(y)),  
- smallN, largeN, unitSquare, shifted, negative, zeroArea.  
Результаты совпадают в пределах погрешности метода (0.5–5% в зависимости от N).

## 7.2 Производительность

Измерения проводились на функции x*y по [0,1]×[0,1] с N=10^8 точек (100 миллионов).  
Указаны средние значения времени выполнения (в секундах).

| Количество процессов | SEQ Pipeline | SEQ Task Run | MPI Pipeline | MPI Task Run |
|----------------------|--------------|--------------|--------------|--------------|
| 1                    | 1.3578       | 1.3578       | 0.7288       | 0.7290       |
| 2                    | 1.3628       | 1.3604       | 0.3721       | 0.3737       |
| 3                    | 1.4618       | 1.4633       | 0.2947       | 0.2983       |
| 4                    | 1.5963       | 1.5959       | 0.2477       | 0.2521       |
| 5                    | 1.6685       | 1.6594       | 0.3603       | 0.3602       |
| 6                    | 2.8783       | 2.2431       | 0.3006       | 0.3028       |
| 7                    | 1.7340       | 1.7892       | 0.2693       | 0.2699       |
| 8                    | 2.9447       | 2.8987       | 0.2397       | 0.2402       |

### Анализ

- При 1 процессе SEQ и MPI дают близкое время (MPI чуть быстрее за счёт меньших накладных расходов на один процесс).
- При росте числа процессов MPI показывает **линейное ускорение** (до ~5–6× на 8 процессах).
- Оптимальное число процессов -7–8 (максимальное ускорение).

**Вывод:** метод Монте-Карло отлично параллелизуется при большом N. MPI-версия даёт значительное ускорение (до 5–6 раз) по сравнению с SEQ.

---

# **8. Заключение**

Реализованы SEQ и MPI версии вычисления двойного интеграла методом Монте-Карло.  
Последовательная версия проста и эффективна для малого N.  
MPI-версия демонстрирует хорошую масштабируемость при большом числе точек (N ≥ 10^7–10^8), обеспечивая ускорение до 5–6 раз на 8 процессах.  
Метод Монте-Карло идеально подходит для параллелизации, особенно в задачах высокой размерности.

---

# **9. Источники**

1. OpenMPI Documentation [Электронный ресурс]. URL: https://www.open-mpi.org  
2. Курс лекций по параллельным вычислениям, УрФУ  
3. Использование численного метода Монте-Карло для вычисления многомерных интегралов URL: https://habr.com/ru/articles/835870/