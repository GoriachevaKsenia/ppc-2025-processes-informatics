
# Global Search Algorithm (Strongin) for One-Dimensional Optimization Problems  Parallelization by Interval Characteristics (SEQ + MPI)

- Student: Sizov Dmitry Igorevich, group 3823Б1ФИ2  
- Technology: SEQ | MPI  
- Variant: 11  

---

## 1. Introduction

The objective of this laboratory work is to study and implement **Strongin’s global
search algorithm** for one-dimensional optimization problems and to analyze its
parallelization using the **MPI** technology.

Strongin’s method is a deterministic global optimization algorithm designed to
locate the *global minimum* of a function defined on a closed interval, even when
the function is non-convex, non-smooth, or contains multiple local minima.

Two implementations are provided:

- a **sequential (SEQ)** reference version;
- a **parallel (MPI)** version based on distributing the computation of interval
  characteristics.

The expected outcome is correct convergence to the global minimum for all test
cases, successful execution of functional and performance tests in the PPC
framework, and measurable speedup of the MPI implementation for computationally
intensive scenarios.

---

## 2. Problem Statement

Given a real-valued function

$$
f : [a, b] \rightarrow \mathbb{R}
$$

it is required to find an approximation of the global minimum

$$
x^* = \arg\min_{x \in [a, b]} f(x)
$$

### Input parameters

- `left`, `right` — interval boundaries;
- `accuracy` — stopping threshold ε;
- `reliability` — Strongin reliability parameter r > 0;
- `max_iterations` — iteration limit;
- `func(x)` — objective function.

### Output

- `argmin` — estimated minimizer;
- `value` — function value at the minimizer;
- `iterations` — number of performed iterations;
- `converged` — convergence flag.

### Constraints

- `left < right`;
- function values must be finite at sampling points;
- accuracy $\varepsilon$ > 0.

---

## 3. Baseline Algorithm (Sequential)

The sequential implementation follows the classical **Strongin global search
algorithm** for one-dimensional optimization.

The algorithm operates on a closed interval

$$
[a, b]
$$

and incrementally refines it by sampling new points in order to approximate the
global minimum of the objective function.

The method is **deterministic** and does not require prior knowledge of the
Lipschitz constant. Instead, a **Lipschitz-like estimate** is computed adaptively
from the already sampled function values.

---

### 3.1 Initial Sampling

At the initial step, the function is evaluated at the interval boundaries:

$$
x_0 = a, \quad x_1 = b,
$$

with corresponding values

$$
f(x_0), \quad f(x_1).
$$

These points form the initial ordered set of samples and define the first interval
for further refinement.

---

### 3.2 Estimation of the Lipschitz-Like Constant

At each iteration, the algorithm computes an adaptive estimate

$$
M = r \cdot \max_i
\frac{\lvert f(x_i) - f(x_{i-1}) \rvert}{x_i - x_{i-1}},
$$

where $r > 0$ is the reliability parameter and the maximum is taken over all
current intervals.

If all slopes are zero, a small positive constant is used to avoid division by
zero.

---

### 3.3 Interval Characteristic

For each interval $[x_{i-1}, x_i]$, the Strongin characteristic is computed as

$$
R_i =M \bigl(x_i - x_{i-1}\bigr)+ \frac{\bigl(f(x_i) - f(x_{i-1})\bigr)^2}{M \bigl(x_i - x_{i-1}\bigr)}- 2 \bigl(f(x_i) + f(x_{i-1})\bigr).
$$

Intervals with larger values of $R_i$ are considered more promising for
containing the global minimum.

---

### 3.4 Selection of the Best Interval

The interval with the maximum characteristic is selected:

$$
i^\ast = \arg\max_{i} R_i
$$

and the corresponding interval $[x_{i^\ast-1}, x_{i^\ast}]$ is refined.

---

### 3.5 Generation of a New Sampling Point

A new sampling point is computed as

$$
x_{\text{new}} =\frac{x_{i-1} + x_i}{2}- \frac{f(x_i) - f(x_{i-1})}{2M}.
$$

If this point lies outside the interval $[x_{i-1}, x_i]$, the midpoint is used
instead.

The new point and its function value are inserted into the ordered sample set.

---

### 3.6 Stopping Criterion

The algorithm terminates when

$$
x_i - x_{i-1} \le \varepsilon,
$$

where $\varepsilon$ is the required accuracy, or when the maximum number of
iterations is reached.

---

## 4. Parallelization by Interval Characteristics (MPI)

The MPI implementation uses **parallelization by interval characteristics**,
which directly follows the structure of Strongin’s global search algorithm and
preserves its deterministic behavior.

At each iteration, the algorithm operates on the ordered set of sampled points

$$
\{x_0, x_1, \dots, x_n\},
$$

which defines $n$ intervals $[x_{i-1}, x_i]$.
Only the computation of interval characteristics and the estimation of the
Lipschitz-like constant are parallelized; all other steps follow the sequential
algorithm.

---

### 4.1 Partitioning of Intervals

The total number of intervals $n$ is divided among $P$ MPI processes.
Each process $p$ is assigned a **contiguous subset of intervals**

$$
[x_{k-1}, x_k], \quad k \in \mathcal{I}_p,
$$

where $\mathcal{I}_p$ denotes the index range handled by process $p$.

The partitioning is static and nearly uniform: each process receives either

$$
\lfloor n / P \rfloor \quad \text{or} \quad \lceil n / P \rceil
$$

intervals. This minimizes load imbalance and avoids dynamic scheduling overhead.

---

### 4.2 Local Characteristic Computation

For each assigned interval $[x_{i-1}, x_i]$, a process computes the Strongin
characteristic

$$
R_i =M (x_i - x_{i-1})+ \frac{(f(x_i) - f(x_{i-1}))^2}{M (x_i - x_{i-1})}- 2 \bigl(f(x_i) + f(x_{i-1})\bigr).
$$

Each process selects the maximum $R_i$ within its local subset and stores the
corresponding interval index.

---

### 4.3 Global Maximum Selection

The globally best interval is determined using the collective operation
`MPI_Allreduce` with the `MPI_MAXLOC` operator.

This operation compares local maxima from all processes and selects the interval
with the maximum characteristic value. As a result, **all MPI processes obtain
the same interval index**, ensuring deterministic behavior identical to the
sequential version.

---

### 4.4 Generation and Broadcast of a New Sampling Point

After selecting the best interval, a new sampling point is computed as

$$
x_{\text{new}} =\frac{x_{i-1} + x_i}{2}- \frac{f(x_i) - f(x_{i-1})}{2M}.
$$

If this point lies outside $[x_{i-1}, x_i]$, the midpoint is used instead.

The new point and its function value are broadcast to all MPI processes, which
insert it into their local copies of the ordered sampling set.

---

## 5. Implementation Details

### 5.1 Project Structure

```text
tasks/sizov_d_global_search
├── common
│   └── include
│       └── common.hpp
├── data
│   └── tests.json
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

### 5.2 Key Implementation Aspects

- The reliability parameter is periodically re-estimated to stabilize convergence.
- Invalid (non-finite) function values are safely ignored.
- Identical stopping criteria are used in SEQ and MPI versions.
- The MPI version preserves deterministic behavior.
- Objective functions in functional tests are defined symbolically and evaluated
  via a custom expression parser.
- Test cases, including function definitions, intervals, and reference solutions, are loaded from external JSON files.

---

## 6. Experimental Setup

| Component  | Value                                |
|------------|--------------------------------------|
| CPU        | 12th Gen Intel(R) Core(TM) i5-12450H |
| RAM        | 16 GB                                |
| OS         | Ubuntu 24.04 (DevContainer / WSL2)   |
| Compiler   | GCC 13.3.0 (g++)                     |
| Build type | Release                              |
| MPI        | Open MPI 4.1.6                       |
| Processes  | 1 / 2 / 4 / 8                        |

---

## 7. Results and Discussion

### 7.1 Correctness

Correctness was verified using functional tests defined in `tests.json`.

Both SEQ and MPI implementations:

- converge to valid global minima;
- produce identical results within tolerance;
- pass all functional tests.

---

### 7.2 Performance

Performance tests were executed using a highly oscillatory function:

$$
f(x) =0.002\,x^2+ 5 \sin(30x)+ \sin\!\bigl(200 \sin(50x)\bigr)+ 0.1 \cos(300x)
$$

The goal of performance testing is **not** to assess numerical accuracy of the
found minimum, but to compare the runtime behavior of the SEQ and MPI
implementations under identical computational conditions.

Metrics were collected using `task_run` time.

#### Performance results

 → 1.1 Performance tests parameters:

```cpp
constexpr double kLeft = -5.0;
constexpr double kRight = 5.0;
constexpr double kAccuracy = 1e-4;
constexpr double kReliability = 3.0;
constexpr int kMaxIterations = 50000;
```

→ 1.2 Results:

| Mode | Processes | Time (s) | Speedup | Efficiency |
|------|-----------|----------|---------|------------|
| SEQ  | 1         | 0.682    | 1.00    | N/A        |
| MPI  | 2         | 0.360    | 1.89    | 94.6%      |
| MPI  | 4         | 0.254    | 2.68    | 67.0%      |
| MPI  | 8         | 0.350    | 1.95    | 24.4%      |

---

 → 2.1 Performance tests parameters:

 ```cpp
 constexpr double kLeft = -10.0;
 constexpr double kRight = 10.0;
 constexpr double kAccuracy = 1e-5;
 constexpr double kReliability = 4.0;
 constexpr int kMaxIterations = 100000;
```

 → 2.2 Results:

 | Mode | Processes | Time (s) | Speedup | Efficiency |
 |------|-----------|----------|---------|------------|
 | SEQ  | 1         | 7.500    | 1.00    | N/A        |
 | MPI  | 2         | 4.943    | 1.52    | 76.0%      |
 | MPI  | 4         | 4.146    | 1.81    | 45.3%      |
 | MPI  | 8         | 8.913    | 0.84    | 10.5%      |

---

### Discussion

- MPI shows speedup for sufficiently computationally intensive cases, where
  characteristic evaluation dominates communication overhead.
- Scalability is limited by synchronization points and collective MPI operations.

---

## 8. Conclusions

In this work:

- Strongin’s global search algorithm was implemented in SEQ and MPI versions;
- a mathematically correct parallelization by interval characteristics was achieved;
- both implementations demonstrated identical convergence behavior;

---

## 9. References

1. Strongin R. G. *Numerical Methods in Multiextremal Problems*.  
2. Sergeyev Y. D., Kvasov D. E. *Global Search Algorithms*.  
3. Gergel V. P. *High-Performance Computing for Multiprocessor Systems*.  
4. Wilkinson B., Allen M. *Parallel Programming*.  

---

## Appendix

```cpp
bool SizovDGlobalSearchMPI::RunImpl() {
  const auto& p = GetInput();

  int rank = 0;
  int size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  if (x_.size() < 2U) {
    return false;
  }

  double m = EstimateM(p.reliability, rank, size);

  for (int iter = 0; iter < p.max_iterations; ++iter) {
    iterations_ = iter + 1;

    if ((iter % 10) == 0) {
      m = EstimateM(p.reliability, rank, size);
    }

    const int n = static_cast<int>(x_.size());
    if (n < 2) {
      converged_ = false;
      break;
    }

    const IntervalChar local = ComputeLocalBestInterval(m, rank, size);
    const int best_idx = ReduceBestIntervalIndex(local, n, size);
    if (best_idx < 1) {
      converged_ = false;
      break;
    }

    if (CheckStopByAccuracy(p, best_idx, rank)) {
      if (rank == 0) {
        converged_ = true;
      }
      break;
    }

    BroadcastNewPoint(best_idx, m, p, rank);
  }

  BroadcastResult(rank);

  GetOutput() = Solution{
      .argmin = best_x_,
      .value = best_y_,
      .iterations = iterations_,
      .converged = converged_,
  };
  return true;
}

```
