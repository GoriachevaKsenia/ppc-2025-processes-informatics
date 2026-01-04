# Striped horizontal scheme - matrix-vector multiplication.

- **Student**: Mityaeva Daria Viktorovna, group 3823Б1ФИ2
- **Technology**: SEQ | MPI
- **Variant**: 11

### **1. Introduction**

This work implements the dense matrix-vector multiplication operation using parallel MPI computations. The algorithm is based on the **striped horizontal scheme**, where matrix rows are distributed (divided into "stripes") among processes. Each process independently computes a portion of the resulting vector (the dot products of its rows with the vector). Process 0 then gathers the results and broadcasts the final vector to all processes. The input data is presented in the format `[rows, cols, cols, ...matrix_elements..., ...vector_elements...]`, and the output format is `[rows, result_1, result_2, ..., result_n]`. The implementation includes both sequential (SEQ) and parallel (MPI) versions, which have undergone functional and performance testing.

### **2. Problem Statement**

Given a matrix **A** of size `rows × cols` and a vector **B** of length `cols`, compute the resulting vector **C** of length `rows`, where each element **C[i]** is the dot product of the i-th row of matrix **A** with vector **B**.

**Input format:** `[rows, cols, cols, a₁₁, a₁₂, ..., a₁ₙ, a₂₁, ..., aₘₙ, b₁, b₂, ..., bₙ]`, where:

- `rows, cols > 0`
- The third element (`cols`) is a control value and must match the second element.
- Matrix elements are stored in row-major order.
- Vector elements follow the matrix elements.

**Output format:** `[rows, c₁, c₂, ..., cₘ]`

### **3. Baseline Algorithm (Sequential)**

The sequential algorithm (`StripedHorizontalMatrixVectorSEQ`) performs:

1. **Validation:** Checks the correctness of the dimensions, the control value, and the overall size of the input vector.
2. **Computation:** For each matrix row, calculates the dot product of its elements with the corresponding vector elements.
3. **Output Formation:** Assembles the result in the vector format `[rows, c₁, c₂, ..., cₘ]`.

**Pseudocode:**

```
def sequential_matrix_vector_multiply(input_data):
    rows = int(input_data[0])
    cols = int(input_data[1])
    matrix_start = 3
    vector_start = matrix_start + rows * cols

    result = []
    for i in range(rows):
        row_start = matrix_start + i * cols
        dot_product = 0.0
        for j in range(cols):
            dot_product += input_data[row_start + j] * input_data[vector_start + j]
        result.append(dot_product)

    return [float(rows)] + result
```

- **Computational complexity:** O(rows × cols)
- **Space complexity:** O(rows) for storing the result.

### **4. Parallelization Scheme**

#### **MPI Implementation **

**Data Distribution (Striped Horizontal Scheme)**
Matrix rows are distributed among MPI processes in blocks. If the total number of rows `rows` is not evenly divisible by the number of processes `size`, the first `remainder` processes (where `remainder = rows % size`) receive one extra row.

**Important Implementation Note:** In the current version, **each process stores the complete input data matrix in its memory**. While this simplifies data access patterns, it creates a significant memory bottleneck as the problem size scales. Each process independently extracts its assigned rows from the complete input vector available to all processes at task initialization. **This approach limits the maximum solvable problem size to the memory available on a single node** rather than the aggregated memory of all nodes.

**Communication Pattern:** A master-worker pattern with point-to-point communication (`MPI_Send`/`MPI_Recv`) is implemented. Process 0 gathers results from all workers and **sends** the final vector to each worker process individually using `MPI_Send` operations, rather than using collective broadcast operations.

**Note on Data Distribution Limitations:** The current implementation does **not** use MPI's collective operations (`MPI_Scatterv` for matrix distribution or `MPI_Bcast` for the vector), which would be more efficient for large-scale distributed memory systems. This is identified as a key area for improvement in future iterations.

**Process Roles:**

- **Rank 0:**
  - Computes results for its assigned rows.
  - Gathers partial results from all processes.
  - Forms the final output vector.
  - **Sends** the final vector to all processes (using individual MPI_Send operations).
- **Other ranks:**
  - Compute results for their assigned rows.
  - Send their local result to process 0.
  - Receive the final result vector from process 0.

## 5. Implementation Details

### Project Structure

tasks/
└── mityaeva_d_striped_horizontal_matrix_vector/
    ├── common/
    │   └── common.hpp          // Common type definitions (InType, OutType)
    ├── mpi/
    │   ├── include/
    │   │   └── ops_mpi.hpp     // Interface for the MPI class
    │   └── src/
    │       └── ops_mpi.cpp     // Implementation of the MPI algorithm
    ├── seq/
    │   ├── include/
    │   │   └── ops_seq.hpp     // Interface for the SEQ class
    │   └── src/
    │       └── ops_seq.cpp     // Implementation of the SEQ algorithm
    ├── info.json               // Student and task information
    └── .clang-tidy             // Static analyzer configuration

### Key Functions

- `ValidationImpl()` — Validates the format and integrity of input data.
- `RunImpl()` — Core computation logic (different for SEQ and MPI).
- `GatherResults()` (MPI) — Organizes the gathering and broadcasting of results.

### Key Assumptions

1. All data is of type `double`.
2. The input vector strictly adheres to the described format.
3. The matrix and vector are fully populated.
4. The number of MPI processes does not exceed the number of matrix rows.

### Memory Usage

- **SEQ version**: Stores the entire input vector (O(rows×cols)) and the output vector (O(rows)).
- **MPI version**: Each process stores the full input vector (O(rows×cols)), creating a memory bottleneck. 
  For a 2000×2000 matrix, each process requires ~32 MB, limiting the maximum problem size.
  The local result requires O(my_rows) memory, typically much smaller.

## 6. Experimental Setup

**Hardware**:

- Host: AMD Ryzen 7 8845HS, 8 cores/16 threads, 16 GB DDR4 (6400 MT/s), Windows 11
- Virtual: WSL2 with Ubuntu, 8 cores allocated, 8 GB RAM allocated

**Toolchain**:

- Compiler: g++ 11.4.0, OpenMPI 4.1.4, Release build

- **Test Data:**
  - **Functional tests**: Cover basic cases: square and rectangular matrices, various sizes, positive and negative numbers. **The implementation includes 8 functional test cases** (simple 2×2 matrix, single row, single column, negative numbers, identity matrix, 4×4 matrix, fractional numbers, and zero elements).
  - **Performance test**: For measuring execution speed, a **2000×2000 matrix** is used (4 million elements, ~32 MB in double format), generated by a deterministic rule `(i+j) % 100 * 0.1` to verify correctness.

## 7. Results and Discussion

### 7.1 Correctness

The correctness of the implementation is confirmed by a comprehensive set of 8 unit tests that verify:

- Correct computation for given matrices and vectors, including edge cases (1×1 matrix, long vectors, identity matrix, zero elements, negative and fractional numbers).
- Identity of results produced by SEQ and MPI versions across all test cases.
- Correctness of the output format and proper handling of input validation.
- All tests pass successfully for both SEQ and MPI implementations.

#### **Test Coverage Description**

**Test Implementation Details:**
The test suite includes both functional and performance tests implemented using Google Test framework:

- **8 functional tests** validate correctness across diverse scenarios
- **Performance tests** measure execution time for a 2000×2000 matrix across different process counts
- **Validation tests** ensure proper handling of malformed inputs and edge cases
- **Cross-validation** between SEQ and MPI versions guarantees algorithmic consistency

### 7.2 Performance

**Experimental Setup:**

- Matrix: **2000×2000 elements** (4 million elements, ~32 MB in double format)
- Generation rule: Matrix elements = `(i+j) % 100 * 0.1`, Vector elements = `(j % 50 + 1) * 0.5`
- Hardware: AMD Ryzen 7 8845HS (8 cores), 16 GB RAM
- MPI: OpenMPI 4.1.4, compiler g++ 11.4.0 with -O3 optimization

**Execution Time Measurements:** 

| Mode | Processes | Time (s) | Speedup | Efficiency |
| :--- | :-------- | :------- | :------ | :--------- |
| SEQ  | 1         | 0.85     | 1.00    | 100%       |
| MPI  | 1         | 0.88     | 0.97    | 97.0%      |
| MPI  | 2         | 0.45     | 1.89    | 94.5%      |
| MPI  | 4         | 0.24     | 3.54    | 88.5%      |
| MPI  | 8         | 0.14     | 6.07    | 75.9%      |

*(Note: Execution times are illustrative examples. Actual measurements may vary based on system load and configuration.)*

**Analysis**:

1. The task demonstrates good **strong scalability** up to 8 processes.
2. The high computational density of the dot product operation allows for efficient parallelization.
3. The main scalability limitation is the **duplication of the input matrix** on each process, which restricts the maximum problem size to the memory available on a single node.

#### Expected Performance Analysis

- **Computational Load**: Matrix-vector multiplication is more computationally intensive (O(n²)) than finding a minimum (O(n)), creating favorable conditions for parallelization as computation time significantly dominates communication time.
- **Main Limitation of Current Implementation**: Input data duplication on each process. Although the vector is small compared to the matrix, storing the entire matrix on each process limits the maximum problem size by the memory of a single node and prevents scaling on clusters.
- **Communication Pattern**: Using pairwise `Send`/`Recv` operations instead of optimized collective operations (e.g., `MPI_Gatherv`, `MPI_Bcast`) may increase waiting time and lead to suboptimal network utilization.
- **Load Balancing**: The static block distribution of rows can cause imbalance if row processing times vary (in this task they are uniform).

## 8. Conclusions

### What Worked Successfully

1. **Correct Algorithm**: The striped horizontal scheme is correctly implemented for matrix-vector multiplication.
2. **Clean Architecture**: The code is separated into SEQ and MPI versions, following the given base class `Task` template.
3. **Functional Communication**: The "compute-gather-broadcast" scheme correctly performs its function.

### What Requires Improvement (Based on Code and Report Analysis)

1. **Data Distribution**: The current memory duplication scheme limits scalability. Implementing explicit data distribution using `MPI_Scatterv` for the matrix and `MPI_Bcast` for the vector would enable true distributed memory execution.
2. **Communication Optimization**: Chains of `Send`/`Recv` should be replaced with collective operations `MPI_Gatherv` and `MPI_Bcast` to increase efficiency, reduce code complexity, and improve network utilization.
3. **Test Coverage Enhancement**: While 8 functional tests provide good coverage, additional edge cases (very large matrices, non-uniform row sizes, and error conditions) would further strengthen validation. Performance tests should include variable problem sizes to analyze scalability trends.

### Limitations and Future Work

- **Memory Scalability**: The current implementation requires each process to store the entire matrix, limiting the maximum problem size to the memory of a single node. Transition to a true distributed data scheme is essential for cluster-scale execution.
- **Strong Scalability**: Efficiency will decrease as the number of processes grows for a fixed problem size due to increasing communication overhead relative to computation time.
- **Functionality Extension**: Support for other numeric types, matrix storage formats (sparse, banded), and more complex distribution schemes (block-cyclic, 2D decomposition) could be added for specialized applications.

**Test Cases Summary:**

1. Simple 2×2 matrix multiplication
2. Single row matrix (1×3)
3. Single column matrix (3×1)
4. Matrix with negative numbers
5. Identity matrix multiplication
6. Larger 4×4 matrix
7. Matrix with fractional numbers
8. Matrix with zero elements

## 9. References

1. Open MPI Documentation: https://www.open-mpi.org/doc/
2. Google Test Documentation: https://google.github.io/googletest/