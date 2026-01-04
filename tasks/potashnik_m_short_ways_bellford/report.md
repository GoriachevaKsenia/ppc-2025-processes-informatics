# Поиск кратчайших путей из одной вершины (алгоритм Беллмана-Форда). С CRS формой хранения графа.
- Student: Поташник Максим Ярославович, group 3823Б1ФИ3
- Technology: SEQ | MPI
- Variant: 23

## 1. Introduction
Задача поиска кратчайших путей из одной вершины является одной из базовых задач теории графов. При этом, часто возникает необходимость решать подобные задачи для больших графов, из-за чего большую актуальность обретают алгоритмы, позволяющие выполнить распараллеливание.

Алгоритм Беллмана–Форда обладает высокой вычислительной сложностью, что делает его подходящим кандидатом для распараллеливания. В данной работе исследуется ускорение вычислений при использовании технологии MPI по сравнению с SEQ реализацией.

## 2. Problem statement
Требуется найти кратчайшие расстояния от фиксированной вершины графа до всех остальных вершин с использованием алгоритма Беллмана–Форда. Граф хранится в форме CRS.

### Входные данные:
Ориентированный взвешенный граф, представленный в CRS (Compressed Row Storage) формате.
Граф генерируется детерминированно по заданному числу вершин.
Источником кратчайших путей является вершина с индексом 0.

### Выходные данные:
Вектор целых чисел — кратчайшие расстояния от вершины-источника до всех остальных вершин графа.

### 3. Baseline Algorithm (Sequential)
Для хранения графа в форме CRS реализован класс Graph, который состоит из:
1. массива row_ptr, задающего границы списков смежности вершин;
2. массива col_idx, содержащего номера смежных вершин;
3. массива weights, содержащего веса рёбер.

```
class Graph {
 public:
  int n;                     
  std::vector<int> row_ptr;  
  std::vector<int> col_ind;  
  std::vector<int> weights;  
};
```

В обеих реализациях алгоритма (SEQ и MPI) выполняется релаксация рёбер из вершины:
```
inline void IterateThroughVertex(const Graph &g, int u, const std::vector<int> &dist, std::vector<int> &dist_out) {
  for (int i = g.Begin(u); i < g.End(u); i++) {
    int v = g.col_ind[i];
    int w = g.weights[i];
    int new_dist = dist[u] + w;
    dist_out[v] = std::min(new_dist, dist_out[v]);
  }
}
```

Инициализация: dist[source] = 0, все остальные dist[v] = 1e9
Повторить (n-1) раз:
- Для каждой вершины u с dist[u] != 1e9:
- Для каждого ребра u->v с весом w:
- dist[v] = min(dist[v], dist[u] + w)
Результат: dist[v] содержит кратчайшее расстояние от вершины source до v для каждой вершины v

SEQ Алгоритм:
```
inline void BellmanFordAlgoSeq(const Graph &g, int source, std::vector<int> &dist) {
  dist.assign(n, 1e9);     
  dist[source] = 0;         
  
  for (int i = 0; i < n - 1; i++) { 
    BellmanFordAlgoIterationSeq(g, dist, dist_next);
    dist.swap(dist_next);  
  }
}
```
Итерация алгоритма:
```
inline void BellmanFordAlgoIterationSeq(const Graph &g, const std::vector<int> &dist, std::vector<int> &dist_next) {
  int n = g.n;
  dist_next = dist;
  for (int uidx = 0; uidx < n; uidx++) {
    if (dist[uidx] == 1e9) {
      continue;
    }
    IterateThroughVertex(g, uidx, dist, dist_next);
  }
}
```

### 4. Parallelization Scheme
В параллельной реализации граф полностью реплицируется на каждом процессе, и все процессы хранят одинаковую CRS-структуру графа. Это происходит с помощью операции MPI_BCast.
Распараллеливание достигается за счёт разделения множества вершин на непересекающиеся диапазоны, при этом каждый MPI-процесс выполняет релаксацию рёбер только для своего диапазона вершин - равномерно распределяется вычислительная нагрузка между процессами.
После каждой итерации локальные результаты объединяются с помощью операции MPI_Allreduce.

MPI Алгоритм:
```
inline void BellmanFordAlgoMpi(const Graph &g, int source, std::vector<int> &dist) {
  // Подготовка данных 

  int start = rank * n / size;
  int end = (rank + 1) * n / size;
  
  for (int i = 0; i < n - 1; i++) {
    BellmanFordAlgoIterationMpi(g, dist, dist_next, start, end);
    MPI_Allreduce(dist_next.data(), dist.data(), n, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
  }
}
```
Итерация алгоритма:
```
inline void BellmanFordAlgoIterationMpi(const Graph &g, const std::vector<int> &dist, std::vector<int> &dist_next, int start, int end) {
  dist_next = dist;
  for (int uidx = start; uidx < end; uidx++) {
    if (dist[uidx] == 1e9) {
      continue;
    }
    IterateThroughVertex(g, uidx, dist, dist_next);
  }
}
```

### 5. Experimental Setup
- Hardware/OS: 12th gen Intel(R) Core(TM) i5-12450H, 8 ядер, 16 GB RAM, Windows 11 x64
- Toolchain: compiler, version, build type (Release/RelWithDebInfo)
    - Cmake 3.28.3
    - Компилятор: g++ (Ubuntu 13.3.0-6ubuntu2~24.04) 13.3.0
    - Использовался Docker-контейнер.
    - Режим сборки: Release.
- Data: Для замера производительности использовался граф с 7000 вершинами, генерируемый произвольно (но детерминированно). 

## 6. Results and Discussion

### 6.1 Correctness
Корректность работы проверена с помощью тестов Google Test на графах с кол-вом вершин: 5, 7, 10, 15, 20.

### 6.2 Performance
| Mode        | Count | Time, s | Speedup | Efficiency |
|-------------|-------|---------|---------|------------|
| seq         | 1     | 3.300   | 1.00    | N/A        |
| mpi         | 2     | 2.030   | 1.63    | 81.5%      |
| mpi         | 4     | 1.690   | 1.95    | 48.8%      |

## 7. Conclusions
Эффективность работы mpi версии при 2-х процессах составляет 81.5%, что является хорошим результатом. Эффективность про 4-х процессах - 48.8%, что объясняется накладными расходами на коммуникацию и синхронизацию между процессами.

## 8. References
1. "Параллельное программирование для кластерных систем" ННГУ им. Лобачевского, ИИТММ