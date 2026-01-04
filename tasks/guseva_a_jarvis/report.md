# Jarvis Algorithm

- Student: Гусева Алёна Сергеевна, 3823Б1ФИ2
- Technology: MPI
- Variant: 25

## 1. Introduction
The Jarvis algorithm (also known as the gift-wrapping algorithm) is a method for constructing the convex hull of a set of points in a plane. This project implements a parallel version of this algorithm using MPI (Message Passing Interface) to accelerate the processing of large datasets.

## 2. Problem Statement
Develop a parallel implementation of the Jarvis algorithm for constructing the convex hull of a binary image represented as a vector of 0s and 1s. The input data includes image dimensions (width, height) and a pixel array. The output should be the pixel array of the original size containing only the boundary pixels.

## 3. Baseline Algorithm
The Jarvis algorithm works by iteratively building the convex hull:
1. Find the leftmost-bottommost point (starting point)
2. For each point on the hull, find the next point with the smallest counterclockwise angle
3. Continue until returning to the starting point

- Time Complexity: `O(n*h)` where `n` is the number of points and `h` is the number of hull points

### Sequentional Algorithm
```c++
std::vector<std::pair<int, int>> BuildConvexHull(const std::vector<std::pair<int, int>>& points) {
    if (points.size() <= 3) return points;
    
    std::vector<std::pair<int, int>> hull;
    size_t start_idx = FindLeftmostBottommost(points);
    size_t current = start_idx;
    
    do {
        hull.push_back(points[current]);
        size_t next = FindNextPoint(points, current);
        current = next;
    } while (current != start_idx);
    
    return hull;
}
```

## 4. Parallelization Scheme

### Parallelization Strategy
The parallel implementation follows a divide-and-conquer approach:
1. Local Hull Computation: Each MPI process computes the convex hull of its local subset of points
2. Data Collection: All local hulls are gathered to the root process (rank 0)
3. Global Hull Computation: The root process computes the final convex hull from all collected points
4. Result Broadcast: The final hull is broadcast to all processes


### Communication Pattern
- `MPI_Gather` - Collects local hull sizes from all processes
- `MPI_Gatherv` - Collects actual hull point data from all processes
- `MPI_Bcast` - Broadcasts the final hull to all processes

### Process Role
- **All processes:** Extract points from the input image and compute local convex hulls
- **Root process (rank 0):** Gathers all local hulls, computes the global hull, and broadcasts results
- **Non-root processes:** Send local hulls to root and receive final results

## 5. Implementation Details
### File structure
```
└── tasks/
    └── guseva_a_jarvis/
        ├── common/                                 
        |   └── common.hpp  . . . . . . . . . . . . Common type definitions 
        |
        ├── mpi/
        |   ├── inlude/
        |   |   └── ops_mpi.hpp . . . . . . . . . . MPI task class declaration 
        |   |                                       inheriting from BaseTask
        |   └── src/
        |       └── ops_mpi.cpp . . . . . . . . . . MPI implementation with data 
        |                                           distribution and reduction
        ├── seq/
        |   ├── inlude/
        |   |   └── ops_seq.hpp . . . . . . . . . . SEQ task class declaration 
        |   |                                       inheriting from BaseTask
        |   └── src/
        |       └── ops_seq.cpp . . . . . . . . . . SEQ implementation with data 
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
- `GusevaAJarvisMPI` Class:
    1. Inherits from BaseTask<InType, OutType>
    2. Manages the entire parallel computation workflow
    3. Coordinates MPI communication
- `GusevaAJarvisSEQ` Class:
    1. Inherits from BaseTask<InType, OutType>
    2. Sequentially calculates convex hull

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
    Generated with Python script binary pixel arrays for input and output. Output is calculated with `scipy` and `numpy`. Each  input array has at least 50% of foreground pixels.

## 7. Results and Discussion

### 7.1 Correctness
Correctness was verified by:
- Testing on verified test data
- CI/CD
- Runs with multiple repetitions, test mixing, different env variables, and MPI flags (e.g., `--oversubscribe`)

### 7.2 Performance
Present time, speedup and efficiency. Example table:

| Mode        | Count | Time, s | Speedup | Efficiency |
|-------------|-------|---------|---------|------------|
| seq         | 1     | 1.00301 | 1.00    | N/A        |
| MPI         | 1     | 0.07488 | 13.39   | 13.39%     |
| MPI         | 2     | 0.07146 | 14.04   | 7.02%      |
| MPI         | 3     | 0.06804 | 14.74   | 4.91%      |
| MPI         | 4     | 0.06541 | 15.33   | 3.75%      |


### Bottlenecks Identified:

- **Root Process Overload**: The root process performs the final hull computation on all collected points, becoming a bottleneck for large datasets
- **Communication Overhead**: `MPI_Gatherv` operations become expensive as hull size increase
- **Load Imbalance**: Point distribution across the processes nay be uneven depending in image content


### Scalability Limits:

1. **Memory Constraints**: Root process must store all collected points

2. **Serial Section**: Final hull computation on root process remains sequential

3. **Communication-to-Computation Ratio**: For small hulls, communication dominates computation

## 8. Conclusions
The MPI implementation successfully parallelizes the column sum computation with:
- Correct results matching the `python.scipy` solution
- Significant speedup for large datasets


## 9. References
1. [Open MPI Documentation](https://www.open-mpi.org/doc/)
2. [Jarvis Algorithm for Convex Hull](https://en.wikipedia.org/wiki/Gift_wrapping_algorithm)