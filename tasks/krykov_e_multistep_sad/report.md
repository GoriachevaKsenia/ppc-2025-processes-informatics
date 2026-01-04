# Отчёт по лабораторной работе  
## «Многошаговая схема решения двумерных задач глобальной оптимизации с использованием MPI»  

**Дисциплина:** Параллельное программирование  
**Преподаватель:** Нестеров Александр Юрьевич и Оболенский Арсений Андреевич  
**Студент:** Крюков Егор Федорович 3823Б1ФИ1
**Вариант:** 12


## Введение

Задачи глобальной оптимизации играют важную роль в вычислительной математике и прикладных областях, таких как оптимизация параметров, машинное обучение, физическое моделирование и анализ сложных функций. В отличие от локальной оптимизации, глобальная оптимизация направлена на поиск глобального экстремума функции, что требует исследования всей области допустимых значений.

В рамках данной работы рассматривается многошаговая схема решения двумерных задач глобальной оптимизации, основанная на последовательном разбиении области поиска. 


## Постановка задачи

Требуется найти глобальный минимум двумерной функции f(x, y) на прямоугольной области( x \in [x_{\min}, x_{\max}], ; y \in [y_{\min}, y_{\max}] ).

Алгоритм должен возвращать координаты точки минимума ( (x^*, y^*) ) и значение функции в этой точке. Поиск осуществляется итерационным методом путём адаптивного разбиения области поиска до достижения заданной точности Eps либо до достижения максимального числа итераций.



## Описание алгоритма

В последовательной реализации используется многошаговая схема глобальной оптимизации. На первом шаге рассматривается начальная прямоугольная область поиска. Для каждой области вычисляется значение функции в её центральной точке. На каждой итерации выбирается область с минимальным значением функции в центре. Далее эта область проверяется на достижение заданной точности: если она достигнута, процесс завершается. В противном случае выбранная область делится на две подобласти вдоль более длинной стороны (по оси X или Y). Для новых областей вычисляются значения функции в центральных точках, после чего они добавляются в общий список областей поиска.

Алгоритм повторяется до выполнения критерия остановки. В качестве результата выбирается область с минимальным значением функции, а искомая точка определяется как центр этой области.

## Описание схемы параллельного алгоритма

Параллельная реализация алгоритма выполнена с использованием MPI и основана на распределении областей поиска между процессами. На каждой итерации текущий список областей транслируется рабочим процессам. Каждому процессу назначается подмножество областей, для которых он вычисляет значения функции и определяет локально лучшую область.

С помощью коллективной операции MPI_Allreduce с оператором определяется глобально лучшая область среди всех локальных кандидатов. Далее информация о выбранной области передаётся всем процессам. Процесс с рангом 0 отвечает за модификацию глобального списка областей: проверку критерия остановки и разбиение выбранной области на две новые.


## Условия экспериментов

Среда выполнения: Windows, MPI.
Количество процессов MPI: 2, 4, 6 и 8 (задаётся параметром -n при запуске mpiexec).
Измерение времени — средствами тестового окружения GoogleTest (режимы pipeline и task_run).


### Результаты при 2 процессах

| Режим выполнения | Время (сек) | Комментарий             |
| ---------------- | ----------: | ----------------------- |
| MPI pipeline     |     0.00177 | Параллельная версия     |
| MPI task_run     |     0.00179 | Параллельная версия     |
| SEQ pipeline     |     0.00180 | Последовательная версия |
| SEQ task_run     |     0.00181 | Последовательная версия |

Вывод: при использовании 2 MPI-процессов параллельная версия демонстрирует сопоставимое время выполнения с последовательной, что объясняется накладными расходами на инициализацию и коммуникации.

### Результаты при 4 процессах

| Режим выполнения | Время (сек) | Комментарий             |
| ---------------- | ----------: | ----------------------- |
| MPI pipeline     |     0.00181 | Параллельная версия     |
| MPI task_run     |     0.00176 | Параллельная версия     |
| SEQ pipeline     |     0.00183 | Последовательная версия |
| SEQ task_run     |     0.00184 | Последовательная версия |

Вывод: при 4 процессах параллельная версия начинает демонстрировать небольшой выигрыш по времени по сравнению с последовательной реализацией. 

### Результаты при 6 процессах

| Режим выполнения | Время (сек) | Комментарий             |
| ---------------- | ----------: | ----------------------- |
| MPI pipeline     |     0.00221 | Параллельная версия     |
| MPI task_run     |     0.00133 | Параллельная версия     |
| SEQ pipeline     |     0.00185 | Последовательная версия |
| SEQ task_run     |     0.00186 | Последовательная версия |

Вывод: при 6 MPI-процессах достигается наилучшее ускорение. Особенно заметен выигрыш в режиме `task_run`.

### Результаты при 8 процессах

| Режим выполнения | Время (сек) | Комментарий             |
| ---------------- | ----------: | ----------------------- |
| MPI pipeline     |     0.00224 | Параллельная версия     |
| MPI task_run     |     0.00190 | Параллельная версия     |
| SEQ pipeline     |     0.00187 | Последовательная версия |
| SEQ task_run     |     0.00188 | Последовательная версия |

Вывод: при 8 процессах выигрыш от параллелизма снижается. Накладные расходы на синхронизацию и обмен данными между процессами начинают компенсировать преимущества распараллеливания. К тому же ухудшение по времени связано с ограничениями процессора


## Заключение

В ходе выполнения лабораторной работы были реализованы последовательный и параллельный алгоритмы решения двумерной задачи глобальной оптимизации на основе многошаговой схемы разбиения области поиска. Последовательная версия алгоритма обеспечивает корректный поиск глобального минимума при заданной точности, а параллельная версия позволяет ускорить вычисления за счёт распределения областей между MPI-процессами.

Параллельная реализация демонстрирует наибольшую эффективность на среднем количестве потоков. Однако из-за накладных расходов она не дает значительного выигрыша в сравнении с последовательной версией


## Источники

1. Лекции Сысоева Александра Владимировича
2. Практические занятия Нестерова Александра Юрьевича и Оболенского Арсения Андреевича
3. Документация по курсу
4. Различные источники из интернета


## Приложение. Исходный код

### SEQ-версия

```cpp
namespace krykov_e_multistep_sad {

namespace {

constexpr double kEps = 1e-4;
constexpr int kMaxIter = 1000;

double EvaluateCenter(const Function2D &f, Region &r) {
  const double xc = 0.5 * (r.x_min + r.x_max);
  const double yc = 0.5 * (r.y_min + r.y_max);
  r.value = f(xc, yc);
  return r.value;
}

Region SplitRegionX(const Region &r, double xm) {
  return Region{.x_min = r.x_min, .x_max = xm, .y_min = r.y_min, .y_max = r.y_max, .value = 0.0};
}

Region SplitRegionXRight(const Region &r, double xm) {
  return Region{.x_min = xm, .x_max = r.x_max, .y_min = r.y_min, .y_max = r.y_max, .value = 0.0};
}

Region SplitRegionY(const Region &r, double ym) {
  return Region{.x_min = r.x_min, .x_max = r.x_max, .y_min = r.y_min, .y_max = ym, .value = 0.0};
}

Region SplitRegionYTop(const Region &r, double ym) {
  return Region{.x_min = r.x_min, .x_max = r.x_max, .y_min = ym, .y_max = r.y_max, .value = 0.0};
}

}  // namespace

KrykovEMultistepSADSEQ::KrykovEMultistepSADSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

bool KrykovEMultistepSADSEQ::ValidationImpl() {
  const auto &[f, x1, x2, y1, y2] = GetInput();
  return static_cast<bool>(f) && x1 < x2 && y1 < y2;
}

bool KrykovEMultistepSADSEQ::PreProcessingImpl() {
  return true;
}

bool KrykovEMultistepSADSEQ::RunImpl() {
  const auto &[f, x_min, x_max, y_min, y_max] = GetInput();

  std::vector<Region> regions;
  regions.push_back(Region{.x_min = x_min, .x_max = x_max, .y_min = y_min, .y_max = y_max, .value = 0.0});
  EvaluateCenter(f, regions.front());

  for (int iter = 0; iter < kMaxIter; ++iter) {
    for (auto &r : regions) {
      EvaluateCenter(f, r);
    }

    auto best_it = std::ranges::min_element(regions, {}, &Region::value);
    Region best = *best_it;
    regions.erase(best_it);

    double dx = best.x_max - best.x_min;
    double dy = best.y_max - best.y_min;

    if (std::max(dx, dy) < kEps) {
      regions.push_back(best);
      break;
    }

    if (dx >= dy) {
      double xm = 0.5 * (best.x_min + best.x_max);
      Region r1 = SplitRegionX(best, xm);
      Region r2 = SplitRegionXRight(best, xm);
      EvaluateCenter(f, r1);
      EvaluateCenter(f, r2);
      regions.push_back(r1);
      regions.push_back(r2);
    } else {
      double ym = 0.5 * (best.y_min + best.y_max);
      Region r1 = SplitRegionY(best, ym);
      Region r2 = SplitRegionYTop(best, ym);
      EvaluateCenter(f, r1);
      EvaluateCenter(f, r2);
      regions.push_back(r1);
      regions.push_back(r2);
    }
  }

  const auto &best = *std::ranges::min_element(regions, {}, &Region::value);
  double x = 0.5 * (best.x_min + best.x_max);
  double y = 0.5 * (best.y_min + best.y_max);
  GetOutput() = {x, y, best.value};

  return true;
}

bool KrykovEMultistepSADSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace krykov_e_multistep_sad


```

### MPI-версия
```cpp
namespace krykov_e_multistep_sad {

namespace {

constexpr double kEps = 1e-4;
constexpr int kMaxIter = 1000;

double EvaluateCenter(const Function2D &f, Region &r) {
  const double xc = 0.5 * (r.x_min + r.x_max);
  const double yc = 0.5 * (r.y_min + r.y_max);
  r.value = f(xc, yc);
  return r.value;
}

int ComputeStartIndex(int rank, int per_proc, int remainder) {
  return (rank * per_proc) + std::min(rank, remainder);
}

int ComputeEndIndex(int start, int per_proc, int rank, int remainder) {
  return start + per_proc + ((rank < remainder) ? 1 : 0);
}

Region SplitXLeft(const Region &r, double xm) {
  return Region{.x_min = r.x_min, .x_max = xm, .y_min = r.y_min, .y_max = r.y_max, .value = 0.0};
}

Region SplitXRight(const Region &r, double xm) {
  return Region{.x_min = xm, .x_max = r.x_max, .y_min = r.y_min, .y_max = r.y_max, .value = 0.0};
}

Region SplitYBottom(const Region &r, double ym) {
  return Region{.x_min = r.x_min, .x_max = r.x_max, .y_min = r.y_min, .y_max = ym, .value = 0.0};
}

Region SplitYTop(const Region &r, double ym) {
  return Region{.x_min = r.x_min, .x_max = r.x_max, .y_min = ym, .y_max = r.y_max, .value = 0.0};
}

Region FindLocalBest(const Function2D &f, std::vector<Region> &regions, int begin, int end) {
  Region best{.x_min = 0.0, .x_max = 0.0, .y_min = 0.0, .y_max = 0.0, .value = std::numeric_limits<double>::max()};

  for (int i = begin; i < end; ++i) {
    EvaluateCenter(f, regions[i]);
    if (regions[i].value < best.value) {
      best = regions[i];
    }
  }

  return best;
}

bool IsRegionSmallEnough(const Region &r) {
  return std::max(r.x_max - r.x_min, r.y_max - r.y_min) < kEps;
}

void ReplaceWithSplit(std::vector<Region> &regions, const Region &best) {
  const double dx = best.x_max - best.x_min;
  const double dy = best.y_max - best.y_min;

  auto it = std::ranges::remove_if(regions, [&](const Region &r) {
    return r.x_min == best.x_min && r.x_max == best.x_max && r.y_min == best.y_min && r.y_max == best.y_max;
  });
  regions.erase(it.begin(), it.end());

  if (dx >= dy) {
    const double xm = 0.5 * (best.x_min + best.x_max);
    regions.push_back(SplitXLeft(best, xm));
    regions.push_back(SplitXRight(best, xm));
  } else {
    const double ym = 0.5 * (best.y_min + best.y_max);
    regions.push_back(SplitYBottom(best, ym));
    regions.push_back(SplitYTop(best, ym));
  }
}

Region FinalBestRegion(const Function2D &f, std::vector<Region> &regions) {
  for (auto &r : regions) {
    EvaluateCenter(f, r);
  }
  return *std::ranges::min_element(regions, {}, &Region::value);
}

}  // namespace

KrykovEMultistepSADMPI::KrykovEMultistepSADMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

bool KrykovEMultistepSADMPI::ValidationImpl() {
  const auto &[f, x1, x2, y1, y2] = GetInput();
  return static_cast<bool>(f) && (x1 < x2) && (y1 < y2);
}

bool KrykovEMultistepSADMPI::PreProcessingImpl() {
  return true;
}

bool KrykovEMultistepSADMPI::RunImpl() {
  int size = 0;
  int rank = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  const auto &[f, x_min, x_max, y_min, y_max] = GetInput();

  std::vector<Region> regions;
  if (rank == 0) {
    regions.push_back(Region{.x_min = x_min, .x_max = x_max, .y_min = y_min, .y_max = y_max, .value = 0.0});
    EvaluateCenter(f, regions.front());
  }

  int stop_flag = 0;

  for (int iter = 0; (iter < kMaxIter) && (stop_flag == 0); ++iter) {
    int n_regions = (rank == 0) ? static_cast<int>(regions.size()) : 0;
    MPI_Bcast(&n_regions, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (n_regions == 0) {
      stop_flag = 1;
      MPI_Bcast(&stop_flag, 1, MPI_INT, 0, MPI_COMM_WORLD);
      break;
    }

    std::vector<Region> local_regions(n_regions);
    if (rank == 0) {
      std::ranges::copy(regions, local_regions.begin());
    }

    MPI_Bcast(local_regions.data(), n_regions * static_cast<int>(sizeof(Region)), MPI_BYTE, 0, MPI_COMM_WORLD);

    const int per_proc = n_regions / size;
    const int remainder = n_regions % size;
    const int begin = ComputeStartIndex(rank, per_proc, remainder);
    const int end = ComputeEndIndex(begin, per_proc, rank, remainder);

    const Region local_best = FindLocalBest(f, local_regions, begin, end);

    struct {
      double value;
      int rank;
    } local_val{.value = local_best.value, .rank = rank}, global_val{};

    MPI_Allreduce(&local_val, &global_val, 1, MPI_DOUBLE_INT, MPI_MINLOC, MPI_COMM_WORLD);

    Region global_best = local_best;
    MPI_Bcast(&global_best, static_cast<int>(sizeof(Region)), MPI_BYTE, global_val.rank, MPI_COMM_WORLD);

    if (rank == 0) {
      stop_flag = static_cast<int>(IsRegionSmallEnough(global_best));
      if (stop_flag == 0) {
        ReplaceWithSplit(regions, global_best);
      }
    }

    MPI_Bcast(&stop_flag, 1, MPI_INT, 0, MPI_COMM_WORLD);
  }

  double x = 0.0;
  double y = 0.0;
  double value = 0.0;

  if (rank == 0) {
    const Region best = FinalBestRegion(f, regions);
    x = 0.5 * (best.x_min + best.x_max);
    y = 0.5 * (best.y_min + best.y_max);
    value = best.value;
  }

  MPI_Bcast(&x, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(&y, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(&value, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  GetOutput() = {x, y, value};
  return true;
}

bool KrykovEMultistepSADMPI::PostProcessingImpl() {
  return true;
}

}  // namespace krykov_e_multistep_sad


```