#include "krykov_e_multistep_sad/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

#include "krykov_e_multistep_sad/common/include/common.hpp"

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
