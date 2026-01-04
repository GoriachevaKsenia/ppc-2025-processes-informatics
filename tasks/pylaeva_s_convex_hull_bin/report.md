# Построение выпуклой оболочки для компонент бинарного изображения

-   Студент: Пылаева Светлана Алексеевна, группа 3823Б1ФИ3
-   Технология: SEQ, MPI
-   Вариант: 20

## 1. Введение

Обработка и анализ изображений являются важными направлениями в области компьютерного зрения, машинного обучения и робототехники. Одной из ключевых задач при работе с изображениями является выделение и анализ связных областей (компонент). В данной работе реализованы последовательная и параллельная версии задачи построения выпуклой оболочки для компонент бинарного изображения. Для распараллеливания вычислений использована технология MPI (Message Passing Interface).

## 2. Постановка задачи

### Задача:

Для каждого связного компонента белых пикселей (значение 255) в бинарном изображении построить выпуклую оболочку.

### Входные данные:

-   изображение в формате `ImageData` (ширина, высота, вектор пикселей).

```cpp
struct ImageData {
  int width{0};
  int height{0};
  std::vector<uint8_t> pixels;
  std::vector<std::vector<Point>> components;
  std::vector<std::vector<Point>> convex_hulls;
}
```

### Выходные данные:

Изображение с сохранёнными исходными размерами и пикселями, дополненное:

-   списком компонент связности.
-   списком выпуклых оболочек для каждой компоненты.

### Ограничения:

-   изображение должно быть корректных размеров (width > 0, height > 0).
-   количество пикселей должно соответствовать формуле `width * height`.
-   выпуклая оболочка строится методом Грэхема.
-   компоненты, содержащие менее 3 точек, не обрабатываются алгоритмом Грэхема (возвращаются как есть).

## 3. Описание последовательного алгоритма

1. `ValidationImpl()` - проверка корректности входных данных (размеры, объём данных).
2. `PreProcessingImpl()` - бинаризация изображения.
3. `RunImpl()` - основной этап обработки.
    - Поиск компонент связности: `FindConnectedComponents()`.
    - Очистка предыдущих результатов.
    - Обработка каждой компоненты:
        - Если компонента содержит ≥3 точек: обработка алгоритмом Грэхема`GrahamScan`.
        - Иначе: сохраняется как есть.
    - Сохранение результатов.
4. `PostProcessingImpl()` - завершающий этап, дополнительных операций не выполняется.

Алгоритм Грэхема:

-   Выбор точки с минимальной Y (и минимальной X при равенстве).
-   Сортировка остальных точек по полярному углу относительно выбранной.
-   Построение оболочки стековым методом.

Реализация последовательной версии алгоритма представлена в Приложение.1

## 4. Описание схемы параллельного алгоритма

**Идея параллелизации:**
Изображение разбивается по строкам между процессами. Каждый процесс обрабатывает свой вертикальный блок, находя компоненты связности в своей области. Затем результаты собираются на процессе 0, где строятся выпуклые оболочки.

**Распределение данных:**

-   Процесс 0 выполняет бинаризацию всего изображения.
-   Изображение разбивается по строкам между процессами с помощью MPI_Scatterv.
-   Каждый процесс получает блок строк `[start_row, end_row)`.
-   Для обработки компонент, пересекающих границы блоков, используется расширенная область с обменом граничными строками между соседними процессами.

**Ранжирование ролей:**

Процесс 0:

-   Бинаризация всего изображения.
-   Распределение данных по процессам (MPI_Scatterv).
-   Обработка своего блока строк.
-   Приём выпуклых оболочек от других процессов.
-   Построение выпуклых оболочек для всех компонент.

Остальные процессы:

-   Получение своего блока данных.
-   Обработка своего блока строк.
-   Отправка выпуклых оболочек процессу 0.

**Ключевые особенности:**

-   Асинхронные вычисления: Каждый процесс независимо находит компоненты связности и строит выпуклые оболочки для своего блока.
-   Обработка граничных компонент: Используется расширенная область (+1 строка сверху и снизу) и обмен граничными строками между соседними процессами.
-   Эффективная коммуникация: Используется MPI_Scatterv для распределения данных и MPI_Gather для сбора результатов.
-   Балансировка нагрузки: Распределение строк происходит с учетом остатка для равномерной загрузки процессов.

Реализация параллельной версии алгоритма представлена в Приложение.2

## 5. Детали реализации

**Файловая структура:**

pylaeva_s_convex_hull_bin/  
├── common/include/common.hpp  
├── mpi/include/ops_mpi.hpp  
├── mpi/src/ops_mpi.cpp  
├── seq/include/ops_seq.hpp  
├── seq/src/ops_seq.cpp  
├── tests/functional/main.cpp  
└── tests/performance/main.cpp

**Ключевые файлы:**

1.  Последовательная реализация (`seq`):

    -   `ops_seq.hpp` - объявление класса `PylaevaSConvexHullBinSEQ`
    -   `ops_seq.cpp` - реализация методов

2.  MPI реализация (`mpi`):

    -   `ops_mpi.hpp` - объявление класса `PylaevaSConvexHullBinMPI`
    -   `ops_mpi.cpp` - реализация методов

3.  Общие компоненты (`common`):

    -   `common.hpp` - общие типы данных и константы

**Ключевые классы и методы:**

Ключевые классы:

-   `PylaevaSConvexHullBinSEQ` - последовательная реализация.
-   `PylaevaSConvexHullBinMPI` - параллельная реализация.

Вспомогательные методы SEQ реализации:

-   `FindConnectedComponents()` — поиск компонент.
-   `GrahamScan()` — построение выпуклой оболочки.

Вспомогательные методы MPI реализации:

-   `ScatterDataAndDistributeWork()` — распределение данных между процессами.
-   `ExchangeBoundaryRows()` — обмен граничными строками между процессами.
-   `ProcessExtendedRegion()` — обработка расширенной области с граничными строками.
-   `ProcessExtendedNeighbors()` — обработка соседей в расширенной области.
-   `FilterLocalComponents()` — фильтрация компонент, принадлежащих текущему процессу.
-   `GatherConvexHullsToRank0()` — сбор выпуклых оболочек на процессе 0.
-   `ReceiveHullsFromProcess()` — приём выпуклых оболочек.
-   `SendHullsToRank0()` — отправка выпуклых оболочек.

## 6. Экспериментальная установка

-   CPU: Intel Core i5-10210U @ 1.60GHz (4 cores, 8 threads)
-   RAM: 16 GB
-   OS: Windows 10 version 22H2
-   Компилятор: MinGW-w64 GCC, C++20, тип сборки Release
-   MPI: Microsoft MPI 10.1.12498.52
-   CMake: 3.30.3
-   Данные:

    1. **Func_tests**  
       7 различных паттернов:

        - квадрат
        - треугольник
        - круг
        - несколько компонент
        - линии
        - отверстие
        - L-форма

    2. **Perf_tests**  
       Перформанс-тесты: сложный паттерн 1000×1000 пикселей с кругами, сетками, прямоугольниками и диагоналями.

### Управление процессами

PPC_NUM_PROC: устанавливается через параметр -np в mpiexec

```cpp
//Запуск с различным количеством процессов MPI
mpiexec -np 1 ./build/bin/ppc_perf_tests --gtest_filter="*PylaevaSConvexHullBinPerfTests*"
mpiexec -np 2 ./build/bin/ppc_perf_tests --gtest_filter="*PylaevaSConvexHullBinPerfTests*"
mpiexec -np 4 ./build/bin/ppc_perf_tests --gtest_filter="*PylaevaSConvexHullBinPerfTests*"
```

## 7. Результаты экспериментов и выводы

Ниже приведена таблица с результатами Perf тестов.

_результаты были получены при запуске build/bin/ppc_perf_tests на 1, 2, 4 и 8 процессах_

| Mode | Count | Time, s | Speedup | Efficiency |
| ---- | ----- | ------- | ------- | ---------- |
| seq  | 1     | 0.079   | 1.00    | N/A        |
| mpi  | 2     | 0.040   | 1.97    | 98%        |
| mpi  | 4     | 0.031   | 2.54    | 63%        |
| mpi  | 8     | 0.022   | 3.59    | 44%        |

**Анализ результатов:**

-   Параллельная MPI-реализация демонстрирует ускорение по сравнению с последовательной версией.
-   Эксперименты показали, что оптимальное число процессов для данной конфигурации — 8. На этом уровне достигнуто ускорение в 3.59 раз по сравнению с последовательной версией.

## 8. Заключение

В ходе работы были реализованы и протестированы последовательная и параллельная версии задачи построения выпуклой оболочки для компонент бинарного изображени с использованием технологии MPI.

**Основные результаты:**

-   Разработаны корректные последовательная и параллельная версии алгоритма.
-   Реализована схема распараллеливания с разбиением изображения по строкам.
-   Достигнуто значительное ускорение - параллельная реализация демонстрирует ускорение до 3.59 раз на 8 процессах по сравнению с последовательной версией.
-   Достигнуто хорошее значение эффективности - параллельная реализация демонстрирует эффективность до 98% на 2 процессах.

## Источники

1. [Документация OpenMPI](https://www.open-mpi.org/doc/)
2. Сысоев А.В. Курс лекций по параллельному программированию

## Приложение

Приложение.1

```cpp
bool PylaevaSConvexHullBinSEQ::RunImpl() {
  FindConnectedComponents();
  processed_data_.convex_hulls.clear();

  for (const auto &component : processed_data_.components) {
    if (component.size() >= 3) {
      processed_data_.convex_hulls.push_back(GrahamScan(component));
    } else if (!component.empty()) {
      processed_data_.convex_hulls.push_back(component);
    }
  }

  GetOutput() = processed_data_;
  return true;
}

void PylaevaSConvexHullBinSEQ::FindConnectedComponents() {
  int width = processed_data_.width;
  int height = processed_data_.height;
  int total_pixels = width * height;
  std::vector<bool> visited(static_cast<size_t>(total_pixels), false);
  processed_data_.components.clear();

  for (int row_y = 0; row_y < height; ++row_y) {
    for (int col_x = 0; col_x < width; ++col_x) {
      size_t idx = (static_cast<size_t>(row_y) * static_cast<size_t>(width)) + static_cast<size_t>(col_x);
      if (processed_data_.pixels[idx] == 255 && !visited[idx]) {
        ProcessConnectedComponent(col_x, row_y, width, height, processed_data_, visited, processed_data_.components);
      }
    }
  }
}

void ProcessConnectedComponent(int start_x, int start_y, int width, int height, const ImageData &processed_data,
                               std::vector<bool> &visited, std::vector<std::vector<Point>> &components) {
  std::vector<Point> component;
  std::queue<Point> q;
  size_t start_idx = (static_cast<size_t>(start_y) * static_cast<size_t>(width)) + static_cast<size_t>(start_x);
  q.emplace(start_x, start_y);
  visited[start_idx] = true;

  while (!q.empty()) {
    Point p = q.front();
    q.pop();
    component.push_back(p);
    ProcessPixelNeighbors(p, width, height, processed_data, visited, q);
  }

  if (!component.empty()) {
    components.push_back(component);
  }
}

void ProcessPixelNeighbors(const Point &p, int width, int height, const ImageData &processed_data,
                           std::vector<bool> &visited, std::queue<Point> &q) {
  const std::vector<std::pair<int, int>> directions = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};

  for (const auto &dir : directions) {
    int nx = p.x + dir.first;
    int ny = p.y + dir.second;

    if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
      int nidx = (ny * width) + nx;
      if (processed_data.pixels[static_cast<size_t>(nidx)] == 255 && !visited[static_cast<size_t>(nidx)]) {
        visited[static_cast<size_t>(nidx)] = true;
        q.emplace(nx, ny);
      }
    }
  }
}

std::vector<Point> PylaevaSConvexHullBinSEQ::GrahamScan(const std::vector<Point> &points) {
  if (points.size() <= 3) {
    return points;
  }

  std::vector<Point> pts = points;
  size_t n = pts.size();

  size_t min_idx = 0;
  for (size_t i = 1; i < n; ++i) {
    if (pts[i].y < pts[min_idx].y || (pts[i].y == pts[min_idx].y && pts[i].x < pts[min_idx].x)) {
      min_idx = i;
    }
  }
  std::swap(pts[0], pts[min_idx]);

  Point pivot = pts[0];
  std::sort(pts.begin() + 1, pts.end(), [&pivot](const Point &a, const Point &b) {
    int orient = Cross(pivot, a, b);
    if (orient == 0) {
      return ((a.x - pivot.x) * (a.x - pivot.x)) + ((a.y - pivot.y) * (a.y - pivot.y)) <
             ((b.x - pivot.x) * (b.x - pivot.x)) + ((b.y - pivot.y) * (b.y - pivot.y));
    }
    return orient > 0;
  });

  std::vector<Point> hull;
  for (size_t i = 0; i < n; ++i) {
    while (hull.size() >= 2 && Cross(hull[hull.size() - 2], hull.back(), pts[i]) <= 0) {
      hull.pop_back();
    }
    hull.push_back(pts[i]);
  }

  return hull;
}

int Cross(const Point &o, const Point &a, const Point &b) {
  return ((a.x - o.x) * (b.y - o.y)) - ((a.y - o.y) * (b.x - o.x));
}
```

Приложение.2

```cpp
bool PylaevaSConvexHullBinMPI::RunImpl() {
  FindConnectedComponentsMpi();
  ProcessComponentsAndComputeHulls();
  GatherConvexHullsToRank0();
  GetOutput() = local_data_;
  return true;
}


void PylaevaSConvexHullBinMPI::FindConnectedComponentsMpi() {
  int width = local_data_.width;
  int height = local_data_.height;
  int local_rows = end_row_ - start_row_;

  int extended_start_row = std::max(0, start_row_ - 1);
  int extended_end_row = std::min(height, end_row_ + 1);
  int extended_local_rows = extended_end_row - extended_start_row;

  std::vector<uint8_t> extended_pixels(static_cast<size_t>(extended_local_rows) * width);

  for (int row = 0; row < local_rows; ++row) {
    int global_row = start_row_ + row;
    int ext_row = global_row - extended_start_row;
    for (int col = 0; col < width; ++col) {
      size_t local_idx = (static_cast<size_t>(row) * static_cast<size_t>(width)) + static_cast<size_t>(col);
      size_t ext_idx = (static_cast<size_t>(ext_row) * static_cast<size_t>(width)) + static_cast<size_t>(col);
      extended_pixels[ext_idx] = local_data_.pixels[local_idx];
    }
  }

  ExchangeBoundaryRows(width, local_rows, extended_start_row, extended_local_rows, extended_pixels);

  std::vector<bool> visited_extended(static_cast<size_t>(extended_local_rows) * width, false);
  std::vector<std::vector<Point>> all_components;

  ProcessExtendedRegion(width, extended_start_row, extended_local_rows, extended_pixels, visited_extended,
                        all_components);

  FilterLocalComponents(all_components);
}

void PylaevaSConvexHullBinMPI::ProcessComponentsAndComputeHulls() {
  local_data_.convex_hulls.clear();

  for (const auto &component : local_data_.components) {
    if (component.size() >= 3) {
      local_data_.convex_hulls.push_back(GrahamScan(component));
    } else if (!component.empty()) {
      local_data_.convex_hulls.push_back(component);
    }
  }
}

void PylaevaSConvexHullBinMPI::GatherConvexHullsToRank0() {
  std::vector<int> hull_counts(size_, 0);
  int local_hull_count = static_cast<int>(local_data_.convex_hulls.size());
  MPI_Gather(&local_hull_count, 1, MPI_INT, hull_counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (rank_ == 0) {
    std::vector<std::vector<Point>> rank0_hulls = local_data_.convex_hulls;
    local_data_.convex_hulls.clear();

    for (int i = 1; i < size_; ++i) {
      ReceiveHullsFromProcess(i, hull_counts[i]);
    }

    for (const auto &hull : rank0_hulls) {
      local_data_.convex_hulls.push_back(hull);
    }
  } else {
    SendHullsToRank0();
  }
}
```
