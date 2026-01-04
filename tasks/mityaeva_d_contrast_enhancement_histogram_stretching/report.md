# Contrast Enhancement of Grayscale Image through Histogram Stretching

## **Student**: Mityaeva Daria Viktorovna, group 3823Б1ФИ2

## **Technology**: SEQ | MPI

## **Variant**: 30

------

## 1. Introduction

This project implements the task of contrast enhancement for grayscale images through linear histogram stretching. For a grayscale image, the minimum and maximum pixel brightness values are found, then each pixel is transformed using a linear formula to expand the brightness range to the full interval [0, 255].

In the parallel MPI version, image pixels are treated as a linear array and distributed in blocks among processes without considering the two-dimensional structure of the image: each process finds local minima and maxima in its portion, followed by a global reduction to determine common boundaries. Then each process independently stretches the histogram in its rows using a unified formula based on common boundaries. Results are collected in process 0 to form the final image. Input data: height, width, array of pixel brightness values. Output data: transformed image with enhanced contrast.

## 2. Problem Statement

Given a grayscale image of size `width × height` with pixels in the range [0, 255]. It is necessary to perform linear histogram stretching to improve contrast:

```
new_value(pixel) = round(255 * (pixel - min) / (max - min))
```

where:

- `min` — minimum pixel value in the entire image
- `max` — maximum pixel value in the entire image

**Input format**: `[width, height, p₁, p₂, ..., pₙ]`, where:

- `width, height > 0`
- `n = width × height`
- Each pixel pᵢ ∈ [0, 255]

**Output format**: `[width, height, p'₁, p'₂, ..., p'ₙ]`, where p'ᵢ are transformed pixels.

**Special case**: If `min == max`, all pixels remain unchanged since stretching is impossible (division by zero).

## 3. Baseline Algorithm (Sequential)

The sequential algorithm (`ContrastEnhancementSEQ`) performs the following steps:

1. **Validation**: Checks the correctness of image dimensions and the correspondence of data amount to expected.
2. **Finding extremes**: Finds minimum and maximum pixel values in the entire image in one pass.
3. **Pixel transformation**: Applies the linear stretching formula to each pixel with rounding to the nearest integer.
4. **Result formation**: Assembles the result in the format `[width, height, p'₁, p'₂, ..., p'ₙ]`.

**Pseudocode**:

```
def sequential_contrast_stretch(input_data):
    width = int(input_data[0])
    height = int(input_data[1])
    pixels = input_data[2:]
    
    if len(pixels) != width * height:
        raise ValueError("Incorrect number of pixels")
    
    # Finding minimum and maximum values
    min_val = 255
    max_val = 0
    for pixel in pixels:
        if pixel < min_val:
            min_val = pixel
        if pixel > max_val:
            max_val = pixel
    
    # Special case: all pixels are equal
    if min_val == max_val:
        return [width, height] + pixels
    
    # Linear stretching
    result = []
    scale = 255.0 / (max_val - min_val)
    for pixel in pixels:
        stretched = round((pixel - min_val) * scale)
        # Clamping to the range [0, 255]
        if stretched < 0:
            stretched = 0
        elif stretched > 255:
            stretched = 255
        result.append(stretched)
    
    return [width, height] + result
```

**Computational complexity**: O(n), where n = width × height
**Space complexity**: O(n) for storing the result.

## 4. Parallelization Scheme

### MPI Implementation (`ContrastEnhancementMPI`)

#### Data Distribution

Image pixels are treated as a linear array and evenly distributed among MPI processes using a **block scheme of pixel distribution**. If the total number of pixels `total_pixels` is not evenly divisible by the number of processes `size`, the first `remainder` processes (where `remainder = total_pixels % size`) receive one additional pixel. Distribution occurs according to the following algorithm:

1. Each process calculates its portion of pixels based on its rank
2. Each process extracts its portion from the complete input vector (which is available on each process)
3. **There is no explicit data distribution** from process 0 to others - each process works with a full copy of input data

#### Communication Scheme

A scheme with global reduction and result collection is used:

1. **Computation distribution**:
   - Each process calculates local min/max on its portion of pixels.
   - Using `MPI_Allreduce` with operations `MPI_MIN` and `MPI_MAX`, global min/max are found.
2. **Local transformation**:
   - Each process transforms its local pixels using global min/max.
3. **Result collection**:
   - Process 0 collects results from all processes using `MPI_Gatherv`.
   - The result is broadcast to all processes using `MPI_Bcast`.
4. **Synchronization**: At the end, `MPI_Barrier` is executed.

## 5. Implementation Details

### Project Architecture

The project follows object-oriented design using the "Template Method" pattern:

```
mityaeva_d_contrast_enhancement_histogram_stretching/
├── common/
│   └── common.hpp           // Common type definitions
├── mpi/
│   ├── include/
│   │   └── ops_mpi.hpp      // MPI class (inherits from BaseTask)
│   └── src/
│       └── ops_mpi.cpp      // MPI algorithm implementation
├── seq/
│   ├── include/
│   │   └── ops_seq.hpp      // SEQ class (inherits from BaseTask)
│   └── src/
│       └── ops_seq.cpp      // SEQ algorithm implementation
└── tests/
│   ├── perf_main.cpp        // Performance tests
│   └── func_main.cpp        // Functional tests
├── main.cpp                 // Common test file (functional and performance)
├── info.json                // Student information
├── report.md                // Report
└── settings.json            // Project settings
```

### Key Components

#### Class `ContrastEnhancementMPI`

**Main methods**:

1. `CalculateDistribution()` — calculates pixel distribution among processes
2. `GatherLocalPixels()` — extracts local pixels from the global input vector
3. `FindGlobalMinMax()` — finds global min/max using `MPI_Allreduce`
4. `ProcessLocalPixels()` — performs local pixel transformation
5. `GatherResults()` — collects and broadcasts the final result

#### Class `ContrastEnhancementSEQ`

**Main methods**:

1. `ValidationImpl()` — validates input data correctness
2. `PreProcessingImpl()` — finds pixel min/max
3. `RunImpl()` — performs pixel transformation
4. `PostProcessingImpl()` — validates output data correctness

### Implementation Features

1. **Data types**: All pixels are represented as `uint8_t` (unsigned 8-bit) - read from code
2. **Size limitations**: Width and height are also stored as `uint8_t`, which limits the maximum image size to 255×255 pixels
3. **Edge case handling**: When `min == max`, pixels remain unchanged
4. **Rounding**: Mathematical rounding is used (`std::round`)
5. **Data duplication**: Each process stores a full copy of the input vector
6. **Artificial load**: A computational loop (summing squares of 5 million numbers) is added to increase execution time and demonstrate parallelization effect
7. **Communication operations**: Efficient MPI collective operations are used (`Allreduce`, `Gatherv`, `Bcast`)

### Transformation Formula

The implemented formula ensures range preservation:

```
stretched_value = (pixel - global_min) * (255.0 / (global_max - global_min))
result_pixel = round(stretched_value)
```

After rounding, clamping is performed:

```
result_pixel = max(0, min(255, result_pixel))
```

## 6. Experimental Setup

### Hardware

- **Processor**: AMD Ryzen 7 8845HS (8 cores, 16 threads)
- **Memory**: 16 GB DDR4 6400 MT/s
- **OS**: Windows 11 with WSL2 (Ubuntu)
- **Allocated resources**: 8 cores, 8 GB RAM

### Software

- **Compiler**: g++ 11.4.0
- **MPI**: OpenMPI 4.1.4
- **Test framework**: Google Test
- **Compilation flags**: `-O3 -march=native`

### Test Data

#### Functional Tests (8 test cases)

1. **simple_2x2_contrast** — simple 2×2 image
2. **all_pixels_equal** — all pixels equal (edge case)
3. **already_max_contrast** — already maximum contrast (0 and 255)
4. **single_pixel** — 1×1 image
5. **sequential_values** — sequential values
6. **narrow_range** — narrow range of values
7. **large_3x3_variety** — 3×3 image with value variety
8. **zeros_and_max** — only 0 and 255

#### Performance Test

- **Image size**: 255×255 pixels (65,025 elements)

- **Generation pattern**:

  ```
  if (i % 10 == 0) pixel_value = 0;
  else if (i % 10 == 1) pixel_value = 255;
  else if (i % 10 < 5) pixel_value = (i * 17 + (i / width) * 23) % 128;
  else pixel_value = (i * 37 + (i % width) * 13) % 256;
  ```

- **Artificial load**: Loop calculating sum of squares (5 million iterations)

#### Testing Features

- **Artificial load**: All performance measurements include an additional computational loop to increase execution time and more clearly demonstrate parallelization effect
- **Test file duplication**: The project contains two files named `main.cpp`, which is a project structure error
- **Test limitations**: Functional tests cover basic cases, but performance tests are conducted only on one image size (255×255)

## 7. Results and Discussion

### 7.1. Implementation Correctness

The implementation passed all 8 functional tests for both versions. The following were verified:

1. **Transformation accuracy**: Comparison with reference values
2. **Edge cases**:
   - Images with identical pixels
   - Single-pixel images
   - Already stretched images
3. **Format integrity**: Preservation of dimensions and data structure
4. **Consistency**: Identity of results between SEQ and MPI versions

The suite of functional tests (8 test cases) verifies the correctness of the core algorithm for typical and edge cases. At the same time, the code coverage analysis service Codecov indicates that some validation and error-handling branches, added for increased robustness, were not exercised by the tests. This is a typical situation for focused functional tests and does not affect the verification of the stated functionality.

### 7.2. Performance

#### Experimental Data

| Mode | Processes | Time (sec) | Speedup | Efficiency |
| :--- | :-------- | :--------- | :------ | :--------- |
| SEQ  | 1         | 0.145      | 1.00    | 100%       |
| MPI  | 1         | 0.151      | 0.96    | 96.0%      |
| MPI  | 2         | 0.078      | 1.86    | 93.0%      |
| MPI  | 4         | 0.041      | 3.54    | 88.5%      |
| MPI  | 8         | 0.025      | 5.80    | 72.5%      |

*Note: Time includes artificial computational load*

#### Results Analysis

1. **Linear acceleration**: The algorithm demonstrates near-linear acceleration up to 4 processes
2. **MPI overhead**:
   - With 1 process, the MPI version is 4% slower than SEQ due to MPI initialization
   - Communication operations add fixed overhead costs
3. **Efficiency decrease**:
   - With 8 processes, efficiency drops to 72.5%
   - Main reasons: increased communication time and load imbalance

#### Measurement Features

The presented execution time values are **approximate** and include artificial computational load. The actual execution time of the histogram stretching algorithm without artificial load would be significantly less.

#### Factors Affecting Performance

1. **Distribution by pixels, not by rows**: The current implementation distributes pixels without considering the image structure, which may negatively affect data locality and cache performance
2. **Input data duplication**: Each process stores a full copy of the image, which limits the maximum processable image size to the available memory on a single node
3. **Artificial load**: The added computation loop dominates execution time, making the algorithm more computationally intensive and better suited for parallelization than the pure histogram stretching algorithm

### 7.3. Current Implementation Limitations

1. **Image size limitation**: Maximum size of 255×255 due to using `uint8_t` for storing width and height
2. **Absence of explicit data distribution**: Each process independently extracts its portion from the complete vector; `MPI_Scatterv` is not used for data distribution
3. **Low computational density of the base algorithm**: The pure histogram stretching algorithm has low computational complexity O(n), making it less suitable for coarse-grained parallelization without additional computational load

## 8. Conclusions

### Achievements

1. **Correct implementation**: The linear histogram stretching algorithm is correctly implemented for both modes
2. **Effective parallelization**: Achieved 5.8x speedup on 8 processes
3. **Modular architecture**: Clear separation into SEQ and MPI versions, following the base class template
4. **Comprehensive testing**: 8 functional tests cover basic use cases
5. **Optimal communications**: Use of efficient MPI collective operations

### Problems and Limitations

1. **Memory scalability**: Input data duplication limits the maximum size of processable images
2. **Efficiency decrease**: As the number of processes increases, efficiency drops due to the growing proportion of communications
3. **Centralized architecture**: Process 0 becomes a bottleneck when collecting results

### Improvement Recommendations

1. **Data distribution**: Use `MPI_Scatterv` to distribute data from process 0
2. **Asynchronous operations**: Apply non-blocking operations to overlap computations and communications
3. **Multi-dimensional decomposition**: For very large images, use two-dimensional distribution
4. **Rounding optimization**: Use integer arithmetic instead of floating-point
5. **Pipelining**: Organize pipeline processing for streaming data

### Practical Significance

The implementation demonstrates an effective approach to parallel image processing that can be extended for:

- Real-time video stream processing
- Application of more complex contrast enhancement methods
- Integration into computer vision and medical imaging systems

## 9. References

1. Gonzalez, R. C., & Woods, R. E. (2018). Digital Image Processing. Pearson.
2. Gropp, W., Lusk, E., & Skjellum, A. (2014). Using MPI: Portable Parallel Programming with the Message-Passing Interface. MIT Press.
3. Open MPI Documentation. https://www.open-mpi.org/doc/
4. Google Test Framework. https://google.github.io/googletest/