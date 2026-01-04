# Сортировка пузырьком (алгоритм чет-нечетной перестановки) / Bubble Sort (Odd–Even Transposition Algorithm)

- Student: Сизов Дмитрий Игоревич, group 3823Б1ФИ2
- Technology: SEQ + MPI
- Variant: 21

## 1. Introduction

The objective of this work is to implement both sequential and parallel (MPI-based) versions of bubble sort using the Odd–Even Transposition Sort algorithm and to compare their correctness and performance. The task belongs to the class of comparison-based sorting algorithms that rely on pairwise exchanges of adjacent elements. The expected outcome is a fully correct implementation of the sorting procedure, successful execution of all functional and performance tests in the PPC framework, and a demonstrated speedup of the MPI version on large input data.

## 2. Problem Statement

Given: an array of integers (`InType = std::vector<int>`).

Required:

- Sort the array in non-decreasing order.
- Provide two implementations: a sequential version and an MPI-based version.
- Both implementations must operate correctly on all provided tests.

Input/output format:

- Input: `std::vector<int>`
- Output: a sorted `std::vector<int>`

Constraint: empty input is not allowed.

## 3. Baseline Algorithm (Sequential)

The sequential version uses the classical odd–even sort algorithm.  
Odd–even sort operates as a sequence of phases, where each phase compares adjacent elements, but the pairs differ between even and odd phases.

During an **even phase**, the algorithm processes element pairs starting at even indices: (0,1), (2,3), (4,5), and so on.  
For each pair, if the left element is greater than the right one, the two elements are swapped.

During an **odd phase**, the algorithm processes pairs starting at odd indices: (1,2), (3,4), (5,6), etc.  
The same swapping rule applies.

These phases alternate, and due to the regular shift of comparison windows, any elements placed in the wrong order gradually move toward their correct positions.  
The algorithm continues alternating phases until a pass occurs with no swaps, which indicates that the array is fully sorted.

Time complexity: **O(n²)**.

## 4. Parallelization Scheme (MPI)

### 4.1 Data Distribution

At the beginning of the parallel algorithm, the root process (rank 0) broadcasts
the total number of elements `n` using `MPI_Bcast`. This ensures that every MPI
process knows the global problem size before data distribution begins.

To partition the input array among the processes, an auxiliary structure
`ScatterPlan` is constructed. It contains two arrays:

• `counts[i]` — the number of elements assigned to process *i*  
• `displs[i]` — the starting global index of process *i*’s block

The distribution is almost uniform: each process receives either  
`⌊n / p⌋` or `⌊n / p⌋ + 1` elements, with the first `n mod p` processes obtaining
one additional element. This guarantees balanced workload while preserving the
global ordering of elements.

The arrays `counts` and `displs` are then passed to `MPI_Scatterv`, which
distributes contiguous segments of the input array to all processes. As a
result, each process receives a local block that corresponds to a continuous
interval of the global array.

---

### 4.2 Local Computation

After data distribution, each process stores its assigned block in the vector
`local`. No preliminary local sorting is performed: the algorithm implements
the classical odd–even transposition sort, which gradually establishes global
order through coordinated local and boundary operations.

During each global phase, every process performs a **local odd–even pass**.
Only adjacent pairs whose **global indices** match the parity of the current
phase are compared.

To minimize unnecessary work, the local loop advances with a step of two,
performing exactly ⌈local_n / 2⌉ comparisons per phase. This corresponds to the
minimal number of comparisons required by the odd–even scheme.

If a process contains fewer than two elements, it skips the local phase
entirely.

---

### 4.3 Neighbor Communication and Odd–Even Phases

The MPI implementation follows the classical **odd–even transposition sort**,
which is a distributed analogue of bubble sort. The algorithm consists of
exactly `n` global phases, where `n` is the total number of elements in the
array. This number of phases is sufficient to guarantee that every element can
move to its correct global position.

Each phase proceeds as follows:

1. **Phase parity determination**

   The parity of the current phase (`even` or `odd`) determines which global
   index pairs must be compared during this pass. All processes compute the
   parity locally as `phase % 2`.

2. **Local odd–even comparisons**

   Each process performs local compare–exchange operations on adjacent elements
   whose global indices match the phase parity. These operations reproduce the
   local step of bubble sort.

3. **Boundary compare–exchange**

   Because global adjacency may cross process boundaries, a boundary
   compare–exchange is required in some phases. Whether such an exchange is
   needed is determined strictly by the **global index** of the boundary
   elements:

   • for the right boundary, the pair  
     `(last_global_index, last_global_index + 1)`  
     is active if `last_global_index % 2 == phase_parity`;

   • for the left boundary, the pair  
     `(first_global_index − 1, first_global_index)`  
     is active if `(first_global_index − 1) % 2 == phase_parity`.

   If the boundary pair is active and the neighboring process exists, the two
   processes exchange one element using `MPI_Sendrecv`. After the exchange:

   • the left process keeps the minimum of the two values,  
   • the right process keeps the maximum.

   If no valid neighbor exists or the boundary pair is inactive in the current
   phase, the boundary exchange is skipped.

This mechanism precisely reproduces the behavior of bubble sort across process
boundaries: smaller elements move left, larger elements move right.

Repeating these steps for all `n` phases guarantees global sortedness without
any additional synchronization or merging stages.

---

### 4.4 Gathering and Finalization

After completing all odd–even transposition phases, each process holds a local
block that is both internally sorted and globally consistent with its
neighbors.

To assemble the final result, the algorithm uses `MPI_Allgatherv`. Each process
contributes its local block, and all blocks are gathered into a single global
array using the same `counts` and `displs` arrays that were employed during data
distribution.

As a result, **every process obtains a complete copy of the fully sorted
array**. No additional merging or post‐processing is required, because the
odd–even transposition algorithm guarantees correctness after all phases have
been executed.

The resulting array is then stored as the final output of the MPI sorting
procedure.

## 5. Implementation Details

### 5.1 Project Structure

```cpp
tasks/sizov_d_bubble_sort
├── common
│   └── include
│       └── common.hpp
├── data
│   ├── test1.txt
│   ├── test2.txt
│   ├── test3.txt
│   ├── ...
│   └── test40.txt
├── mpi
│   ├── include
│   │   └── ops_mpi.hpp
│   └── src
│       └── ops_mpi.cpp
├── seq
│   ├── include
│   │   └── ops_seq.hpp
│   └── src
│       └── ops_seq.cpp
├── tests
│   ├── functional
│   │   └── main.cpp
│   └── performance
│       └── main.cpp
├── info.json
├── report.md
└── settings.json

```

### 5.2 Data

The functional tests use a set of input files located in:
  `tasks/sizov_d_bubble_sort/data/testN.txt`

Each file contains two lines:  

1. an unsorted array,  
2. the reference sorted result.

### 5.3 Additional Helper Functions

Several auxiliary helper functions support the MPI implementation of the
odd–even transposition sort. All algorithm-specific helpers are defined inside
an anonymous namespace in `mpi/src/ops_mpi.cpp`, which restricts their visibility
to the current translation unit and emphasizes their internal nature.

- **`MakeScatterPlan(n, comm_size)`**  
  Constructs a helper structure `ScatterPlan` containing two arrays:
  `counts` and `displs`.  
  The array `counts[i]` specifies how many elements are assigned to process *i*,
  while `displs[i]` stores the starting global index of that block.

  The input array is partitioned almost uniformly: each process receives either
  `⌊n / p⌋` or `⌊n / p⌋ + 1` elements, and the first `n mod p` processes obtain
  one extra element. This ensures balanced workload distribution while preserving
  the global order of elements.

- **`CompareSwap(a, b)`**  
  A small utility that performs an elementary compare–exchange operation.
  If `a > b`, the values are swapped.  
  This function represents the basic operation of bubble sort and is used
  both in local comparisons and indirectly in boundary updates.

- **`LocalOddEvenPhase(local, global_offset, phase_parity)`**  
  Executes a single local odd–even pass within a process’s block.  
  Only adjacent pairs whose *global indices* match the parity of the current
  phase are compared and potentially swapped.

  To reduce unnecessary work, the function advances the loop with a step of two,
  performing the minimal number of comparisons required by the odd–even scheme.
  This operation is the direct local analogue of one bubble-sort pass.

- **`ExchangeRightIfNeeded(local, plan, rank, comm_size, phase_parity, tag)`**  
  Performs a boundary compare–exchange with the right neighboring process, if
  such a neighbor exists and the boundary pair is active in the current phase.

  The function checks whether the global index pair
  `(last_global_index, last_global_index + 1)` matches the phase parity. If so,
  the two processes exchange a single boundary value using `MPI_Sendrecv`.
  After the exchange, the left process keeps the minimum of the two values,
  ensuring correct bubble-sort semantics across process boundaries.

- **`ExchangeLeftIfNeeded(local, plan, rank, phase_parity, tag)`**  
  Symmetric to `ExchangeRightIfNeeded`, this function handles boundary
  compare–exchange with the left neighbor.

In addition to the MPI helpers, the functional test utilities include:

- **`TrimString(std::string& s)`** *(functional test utility)*  
  Removes leading and trailing whitespace characters (`\r`, `\n`, `\t`, spaces)
  before test input parsing to prevent formatting-related errors.

Each helper isolates a well-defined step of the algorithm, improving the clarity
and maintainability of the overall MPI odd–even transposition sort
implementation.

## 6. Experimental Setup

| Component  | Value                                   |
|------------|-----------------------------------------|
| CPU        | 12th Gen Intel(R) Core(TM) i5-12450H    |
| RAM        | 16 GB                                   |
| ОС         | OS: Ubuntu 24.04 (DevContainer / WSL2)  |
| Compiler   | GCC 13.3.0 (g++), C++20, CMake, Release |
| MPI        | mpirun (Open MPI) 4.1.6                 |

## 7. Results and Discussion

### 7.1 Correctness

Correctness has been verified through:

- multiple functional tests,
- comparison against `std::ranges::sort`,
- matching results between the MPI and SEQ implementations.

### 7.2 Performance

For local performance measurements and for comparing the sequential and parallel versions, a unified input size of 75,000/175,000/275,000 elements was used.  
The performance tests were executed locally on my machine with the hardware configuration listed above.  

The speedup is computed as:

$$S(p) = \frac{T_{seq}}{T_{p}}$$

The efficiency is computed as:

$$E(p) = \frac{S(p)}{p} \cdot 100\%$$

**Performance (pipeline, size = 75,000):**

| Mode | Processes | Time (s)       | Speedup S(p) | Efficiency E(p) |
|------|-----------|----------------|--------------|-----------------|
| SEQ  | 1         | 1.1382211208   | 1.0000       | N/A             |
| MPI  | 2         | 1.4889316424   | 0.7645       | 38.22%          |
| MPI  | 4         | 1.0739249196   | 1.0599       | 26.50%          |
| MPI  | 8         | 1.8178929094   | 0.6261       | 7.83%           |

---

**Performance (task_run, size = 75,000):**

| Mode | Processes | Time (s)       | Speedup S(p) | Efficiency E(p) |
|------|-----------|----------------|--------------|-----------------|
| SEQ  | 1         | 0.2347642899   | 1.0000       | N/A             |
| MPI  | 2         | 1.5365564150   | 0.1528       | 7.64%           |
| MPI  | 4         | 1.1454324088   | 0.2050       | 5.12%           |
| MPI  | 8         | 1.8488088746   | 0.1270       | 1.59%           |

---

**Performance (pipeline, size = 175,000):**

| Mode | Processes | Time (s)       | Speedup S(p) | Efficiency E(p) |
|------|-----------|----------------|--------------|-----------------|
| SEQ  | 1         | 6.4017530918   | 1.0000       | N/A             |
| MPI  | 2         | 7.9087940308   | 0.8090       | 40.45%          |
| MPI  | 4         | 5.8502519922   | 1.0945       | 27.36%          |
| MPI  | 8         | 9.7293114716   | 0.6578       | 8.22%           |

---

**Performance (task_run, size = 175,000):**

| Mode | Processes | Time (s)       | Speedup S(p) | Efficiency E(p) |
|------|-----------|----------------|--------------|-----------------|
| SEQ  | 1         | 1.4572491646   | 1.0000       | N/A             |
| MPI  | 2         | 8.2717706478   | 0.1762       | 8.81%           |
| MPI  | 4         | 5.9481297592   | 0.2449       | 6.12%           |
| MPI  | 8         | 9.3558528120   | 0.1558       | 1.95%           |

---

**Performance (pipeline, size = 275,000):**

| Mode | Processes | Time (s)       | Speedup S(p) | Efficiency E(p) |
|------|-----------|----------------|--------------|-----------------|
| SEQ  | 1         | 14.0848023891  | 1.0000       | N/A             |
| MPI  | 2         | 19.9201253816  | 0.7071       | 35.35%          |
| MPI  | 4         | 12.4546995072  | 1.1309       | 28.27%          |
| MPI  | 8         | 14.0781657308  | 1.0005       | 12.51%          |

---

**Performance (task_run, size = 275,000):**

| Mode | Processes | Time (s)       | Speedup S(p) | Efficiency E(p) |
|------|-----------|----------------|--------------|-----------------|
| SEQ  | 1         | 2.8867703438   | 1.0000       | N/A             |
| MPI  | 2         | 19.5067180478  | 0.1480       | 7.40%           |
| MPI  | 4         | 11.8059592702  | 0.2445       | 6.11%           |
| MPI  | 8         | 12.9787113786  | 0.2224       | 2.78%           |

## 8. Conclusions

Both the sequential (SEQ) and MPI versions of the odd–even bubble sort were fully implemented.

## 9. References

- Гергель В.П. Высокопроизводительные вычисления для многопроцессорных многоядерных систем. Серия «Суперкомпьютерное образование».
- Гергель В.П. Теория и практика параллельных вычислений. – М.: Интуит Бином. Лаборатория знаний, 2007.
- Wilkinson B., Allen M. Parallel programming: techniques and applications using networked workstations and parallel computers. – Prentice Hall, 1999.

## Appendix

```cpp
bool SizovDBubbleSortMPI::RunImpl() {
  int rank = 0;
  int comm_size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &comm_size);

  int n = 0;
  if (rank == 0) {
    n = static_cast<int>(data_.size());
  }
  MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (n <= 1) {
    if (rank != 0) {
      data_.assign(n, 0);
    }
    if (n == 1) {
      MPI_Bcast(data_.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
    }
    GetOutput() = data_;
    return true;
  }

  const ScatterPlan plan = MakeScatterPlan(n, comm_size);
  const int local_n = plan.counts[rank];

  std::vector<int> local(local_n);
  MPI_Scatterv(rank == 0 ? data_.data() : nullptr, plan.counts.data(), plan.displs.data(), MPI_INT, local.data(),
               local_n, MPI_INT, 0, MPI_COMM_WORLD);

  for (int phase = 0; phase < n; ++phase) {
    const int parity = phase & 1;
    const int tag = phase;

    LocalOddEvenPhase(local, plan.displs[rank], parity);
    ExchangeRightIfNeeded(local, plan, rank, comm_size, parity, tag);
    ExchangeLeftIfNeeded(local, plan, rank, parity, tag);
  }

  std::vector<int> result(n);
  MPI_Allgatherv(local.data(), local_n, MPI_INT, result.data(), plan.counts.data(), plan.displs.data(), MPI_INT,
                 MPI_COMM_WORLD);

  GetOutput() = std::move(result);
  return true;
}
```
