# Hypercube topology

- Student: Гусева Алёна Сергеевна, 3823Б1ФИ2
- Technology: MPI
- Variant: 10

## 1. Introduction
A hypercube network topology is a high-dimensional interconnection structure where each node (processor) is connected to others whose binary addresses differ by exactly one bit. This topology provides efficient routing patterns for parallel algorithms, particularly for reduction and broadcast operations, making it suitable for parallel computing systems.

## 2. Problem Statement
The problem involves implementing efficient data routing within a hypercube topology. This requires designing algorithms for point-to-point message passing (`SendRecv`) and global reduction (`ReduceSum`) across hypercube dimensions. This topology is particularly useful in parallel numerical computations where processes need to exchange data following binary-reflected Gray code patterns.
The implementation is verified by calculating a derived finite sum.

## 3. Baseline Algorithm
Finite sum local calculation:
Hypercube routing represented as node-to-node function `SendRecv` and group function `ReduceSum`:
```c++
template <typename _datatype>
  void ReduceSum(_datatype local_value, MPI_Datatype mpi_datatype, _datatype &result_buf, int root = 0) const {
    if (comm_ == MPI_COMM_NULL) {
      return;
    }

    _datatype current_value = local_value;
    for (int dim = 0; dim < ndims_; ++dim) {
      int neighbor = rank_ ^ (1 << dim);

      _datatype received_value = 0;
      MPI_Status status;
      if ((rank_ >> dim) & 1) {
        MPI_Send(&current_value, 1, mpi_datatype, neighbor, dim, comm_);
        MPI_Recv(&received_value, 1, mpi_datatype, neighbor, dim, comm_, &status);
      } else {
        MPI_Recv(&received_value, 1, mpi_datatype, neighbor, dim, comm_, &status);
        MPI_Send(&current_value, 1, mpi_datatype, neighbor, dim, comm_);
      }

      current_value += received_value;
    }
    if (rank_ == root) {
      result_buf = current_value;
    }
  }
  ```
  ```c++
  template <typename _datatype>
  void SendRecv(int sender, int receiver, _datatype message, MPI_Datatype mpi_datatype, _datatype &dst_buff) const {
    if (sender >= size_ || receiver >= size_) {
      throw std::invalid_argument("Sender or receiver is out of hypercube.");
    }
    int current = sender;
    _datatype current_msg = message;

    while (current != receiver) {
      int route = current ^ receiver;
      int dim = 0;
      while ((route & (1 << dim)) == 0) {
        dim++;
      }

      int next = current ^ (1 << dim);

      _datatype recv_msg = 0;
      MPI_Status status;

      if (rank_ == current) {
        MPI_Send(&current_msg, 1, mpi_datatype, next, 0, comm_);
      }
      if (rank_ == next) {
        MPI_Recv(&recv_msg, 1, mpi_datatype, current, 0, comm_, &status);
      }

      current = next;
      current_msg = recv_msg;
    }

    if (rank_ == receiver) {
      dst_buff = current_msg;
    }
  }
```


## 4. Parallelization Scheme

For the finite sum calculation, the summing range is equally divided between nodes. Finally, each node's local result is collected by `ReduceSum`  at the root process (`rank=0`).
Hypercube routing corresponds to the hypercube E-cube routing algorithm.

### Communication Pattern
- **ReduceSum**: Implements hypercube reduction by exchanging and summing partial results across each dimension using the `rank XOR (1 << dim)` neighbor pattern.

- **SendRecv**: Routes messages between any two nodes in the hypercube by traversing dimensions where their binary addresses differ.

### Process Role
Each process in the hypercube:

1. Receives its portion of the data range based on rank and total processes.

2. Computes the local sum of function values for its assigned subrange.

3. Participates in hypercube reduction to accumulate the global sum at the root.

4. Receives the final result via broadcast.

## 5. Implementation Details
### File structure
```
└── tasks/
    └── guseva_a_hypercube/
        ├── common/                                 
        |   └── common.hpp  . . . . . . . . . . . . Common type definitions 
        |
        ├── mpi/
        |   ├── inlude/
        |   |   ├── ops_mpi.hpp . . . . . . . . . . MPI task class declaration 
        |   |   |                                   inheriting from BaseTask
        |   |   |
        |   |   └── hypercube.hpp . . . . . . . . . Hypercube router
        |   |
        |   └── src/
        |       └── ops_mpi.cpp . . . . . . . . . . MPI implementation with data 
        |                                           distribution and reduction
        |
        ├── tests/ 
        |   ├── functional/
        |   |   └── main.cpp
        |   |
        |   └── performance/
        |       └── main.cpp
        |
        ├── info.json
        ├── report.md
        └── settings.json

```
### Key Classes
- **`Hypercube`**: Manages hypercube topology, including creation of hypercube subcommunicator and implementation of `ReduceSum` and `SendRecv` operations

- **`GusevaAHypercubeMine`**: Inherits from `BaseTask`, orchestrates the parallel computation workflow, handles data distribution and hypercube reduction

- **`GusevaARunFuncTestsProcesses`**: Functional test class verifying correctness for various mathematical functions (linear, quadratic, logarithmic, exponential, trigonometric)

- **`GusevaAHypercubePerfTestProcesses`**: Performance test class evaluating efficiency on large datasets

### Important Assumptions
- Hypercube size is limited to powers of two; excess processes are excluded from computatio

- Function pointers provided in input are valid and thread-safe

## 6. Experimental Setup
- **Hardware/OS**: 
    - **Host**: Intel Core i7-14700k, 8+12 cores, 32 Gb DDR4, Windows 10 (10.0.19045.6456) 
    - **Virtual**: Intel Core i7-14700k, 12 cores, 8 Gb, WSL2 (2.6.1.0) + Ubuntu (24.04.3 LTS)
- **Toolchain**:
    | Compiler | Version | Build Type |
    |:-:|:-:|:-:|
    | gcc | 14.2.0 x86_64-linux-gnu | Release
    | clang | 21.1.0 x86-64-pc-linus-gnu | Release
- **Environment**:
    | Variable | Values | Comment |
    |:-|:-:|:-:|
    | PPC_NUM_PROC | [1, 4] | For `srcipts/run_tests.py` single executions w/o `--count` arg |
    | PPC_NUM_THREADS | 1 | Similarly as previous|
    | OMPI_ALLOW_RUN_AS_ROOT | 1 | Needed by OpenMPI to run under Docker and .devcointer |
    | OMPI_ALLOW_RUN_AS_ROOT_CONFIRM | 1 | Confirmation for previous env variable|
- **Data**: 

Test data is based on mathematical sums of finite sequences with known closed-form solutions. Functional tests cover 8 different mathematical functions with ranges up to 1e3 elements, while performance tests evaluate on 1e8 elements with trigonometric functions to avoid `double` overflow.
## 7. Results and Discussion

### 7.1 Correctness
Correctness was verified by:
- Testing with 8 different mathematical functions having known analytical solutions
- CI/CD
- Runs with multiple repetitions, test mixing, different env variables, and MPI flags (e.g., `--oversubscribe`)
- The epsilon constant (`precision = 10e-12`) ensures floating-point precision

### 7.2 Performance
Present time, speedup and efficiency. Example table:

| Mode        | Count | Time, s | Speedup | Efficiency |
|-------------|-------|---------|---------|------------|
| MPI         | 1     | 1.00301 | 1.00    | N/A        |
| MPI         | 2     | 0.7488  | 13.39   | 13.39%     |
| MPI         | 3     | 0.07146 | 14.04   | 7.02%      |
| MPI         | 4     | 0.06804 | 14.74   | 4.91%      |

**Discussion**:

### Bottlenecks Identified:

- **Hypercube Creation Overhead**: Splitting `MPI_COMM_WORLD` and creating hypercube subcommunicator adds initialization cost
- **Power-of-Two Limitation**: Exclusion of processes when total MPI processes is not a power of two reduces resource utilization


### Scalability Limits:

1. **Diminishing Returns**: The speedup curve shows strong diminishing returns:
   - From 1 to 2 processes: +0.65 speedup
   - From 2 to 3 processes: +0.70 speedup  
   - From 3 to 4 processes: +0.59 speedup

2. **Communication-to-Computation Ratio**: For the given problem size, the ratio favors communication over computation. The column sum operation is computationally inexpensive (O(N) per element), making it difficult to amortize communication costs.

## 8. Conclusions
The MPI implementation successfully parallelizes the column sum computation with:
- Correct results matching the analytical math solution
- Significant speedup for large datasets

### Limitations:
- Communication overhead in hypercube reduction limits efficiency for small to medium problem sizes
- Exclusion of processes when total MPI processes is not a power of two
- Fixed data distribution strategy may not be optimal for all function types

## 9. References
1. [Open MPI Documentation](https://www.open-mpi.org/doc/)
2. [Hypercube internetwork topology](https://en.wikipedia.org/wiki/Hypercube_internetwork_topology)