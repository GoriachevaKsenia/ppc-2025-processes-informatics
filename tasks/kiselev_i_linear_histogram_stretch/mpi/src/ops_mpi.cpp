#include "kiselev_i_linear_histogram_stretch/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

#include "kiselev_i_linear_histogram_stretch/common/include/common.hpp"

namespace kiselev_i_linear_histogram_stretch {

KiselevITestTaskMPI::KiselevITestTaskMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;

  if (!in.pixels.empty()) {
    GetOutput().resize(in.pixels.size());
  }
}

bool KiselevITestTaskMPI::ValidationImpl() {
  const auto &img = GetInput();
  return img.width > 0 && img.height > 0 && img.pixels.size() == img.width * img.height;
}

bool KiselevITestTaskMPI::PreProcessingImpl() {
  return true;
}

bool KiselevITestTaskMPI::RunImpl() {
  int rank = 0;
  int size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  constexpr int kRoot = 0;

  std::size_t total_size = 0;
  if (rank == kRoot) {
    total_size = GetInput().pixels.size();
  }

  MPI_Bcast(&total_size, 1, MPI_UNSIGNED_LONG_LONG, kRoot, MPI_COMM_WORLD);

  const std::size_t base = total_size / static_cast<std::size_t>(size);
  const std::size_t extra = total_size % static_cast<std::size_t>(size);

  std::vector<int> counts(size, 0);
  std::vector<int> offsets(size, 0);

  if (rank == kRoot) {
    std::size_t shift = 0;
    const auto size_sz = static_cast<std::size_t>(size);

    for (std::size_t index = 0; index < size_sz; ++index) {
      const std::size_t add = (index < extra) ? 1 : 0;
      counts[static_cast<int>(index)] = static_cast<int>(base + add);
      offsets[static_cast<int>(index)] = static_cast<int>(shift);
      shift += static_cast<std::size_t>(counts[static_cast<int>(index)]);
    }
  }

  int local_count = 0;
  MPI_Scatter(counts.data(), 1, MPI_INT, &local_count, 1, MPI_INT, kRoot, MPI_COMM_WORLD);

  std::vector<unsigned char> local_pixels(static_cast<std::size_t>(local_count));

  MPI_Scatterv(GetInput().pixels.data(), counts.data(), offsets.data(), MPI_UNSIGNED_CHAR, local_pixels.data(),
               local_count, MPI_UNSIGNED_CHAR, kRoot, MPI_COMM_WORLD);

  unsigned char local_min = std::numeric_limits<unsigned char>::max();
  unsigned char local_max = std::numeric_limits<unsigned char>::min();

  for (unsigned char px : local_pixels) {
    local_min = std::min(local_min, px);
    local_max = std::max(local_max, px);
  }

  unsigned char global_min = 0;
  unsigned char global_max = 0;

  MPI_Allreduce(&local_min, &global_min, 1, MPI_UNSIGNED_CHAR, MPI_MIN, MPI_COMM_WORLD);
  MPI_Allreduce(&local_max, &global_max, 1, MPI_UNSIGNED_CHAR, MPI_MAX, MPI_COMM_WORLD);

  if (global_min != global_max) {
    const double scale = 255.0 / static_cast<double>(global_max - global_min);

    for (unsigned char &px : local_pixels) {
      const double value = static_cast<double>(px - global_min) * scale;
      px = static_cast<unsigned char>(std::lround(value));
    }
  }

  MPI_Gatherv(local_pixels.data(), local_count, MPI_UNSIGNED_CHAR, GetOutput().data(), counts.data(), offsets.data(),
              MPI_UNSIGNED_CHAR, kRoot, MPI_COMM_WORLD);

  return true;
}

bool KiselevITestTaskMPI::PostProcessingImpl() {
  return true;
}

}  // namespace kiselev_i_linear_histogram_stretch
