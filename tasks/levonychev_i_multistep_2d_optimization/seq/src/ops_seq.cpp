#include "levonychev_i_multistep_2d_optimization/seq/include/ops_seq.hpp"

#include <algorithm>
#include <cstddef>
#include <limits>
#include <vector>

#include "levonychev_i_multistep_2d_optimization/common/include/common.hpp"
#include "levonychev_i_multistep_2d_optimization/common/include/optimization_common.hpp"

namespace levonychev_i_multistep_2d_optimization {

LevonychevIMultistep2dOptimizationSEQ::LevonychevIMultistep2dOptimizationSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = OptimizationResult();
}

bool LevonychevIMultistep2dOptimizationSEQ::ValidationImpl() {
  const auto &params = GetInput();

  bool is_correct_ranges = (params.x_min < params.x_max) && (params.y_min < params.y_max);
  bool isnt_correct_func = !params.func;
  bool is_correct_params = (params.num_steps > 0) && (params.grid_size_step1 > 0) && (params.candidates_per_step > 0);

  return is_correct_ranges && !isnt_correct_func && is_correct_params;
}

bool LevonychevIMultistep2dOptimizationSEQ::PreProcessingImpl() {
  GetOutput() = OptimizationResult();
  return true;
}

bool LevonychevIMultistep2dOptimizationSEQ::RunImpl() {
  const auto &params = GetInput();
  auto &result = GetOutput();

  SearchRegion current_region(params.x_min, params.x_max, params.y_min, params.y_max);

  for (int step = 0; step < params.num_steps; ++step) {
    int grid_size = params.grid_size_step1 * (1 << step);

    std::vector<Point> local_points;
    size_t total_points = static_cast<size_t>(grid_size) * static_cast<size_t>(grid_size);
    local_points.reserve(total_points);
    SearchInRegion(local_points, params.func, current_region, grid_size, 1);

    std::vector<Point> local_candidates;
    int num_local_candidates = std::min(params.candidates_per_step, static_cast<int>(local_points.size()));
    local_candidates.assign(local_points.begin(), local_points.begin() + num_local_candidates);

    std::vector<Point> all_candidates = local_candidates;

    std::vector<Point> valid_candidates;
    for (const auto &cand : all_candidates) {
      if (cand.value < std::numeric_limits<double>::max()) {
        valid_candidates.push_back(cand);
      }
    }
    std::ranges::sort(valid_candidates, [](const Point &a, const Point &b) { return a.value < b.value; });

    int num_global = std::min(params.candidates_per_step, static_cast<int>(valid_candidates.size()));
    all_candidates.assign(valid_candidates.begin(), valid_candidates.begin() + num_global);

    std::vector<SearchRegion> new_regions;
    for (const auto &cand : all_candidates) {
      double margin_x = (params.x_max - params.x_min) * 0.05 / (1 << step);
      double margin_y = (params.y_max - params.y_min) * 0.05 / (1 << step);

      SearchRegion new_region(std::max(params.x_min, cand.x - margin_x), std::min(params.x_max, cand.x + margin_x),
                              std::max(params.y_min, cand.y - margin_y), std::min(params.y_max, cand.y + margin_y));
      new_regions.push_back(new_region);
    }
    if (!new_regions.empty()) {
      current_region = new_regions[0];
    }
  }

  int power_of_2 = 1;
  for (int i = 0; i < params.num_steps - 1; ++i) {
    power_of_2 *= 2;
  }
  int final_grid_size = params.grid_size_step1 * power_of_2;
  std::vector<Point> local_points;
  local_points.reserve(static_cast<size_t>(final_grid_size) * static_cast<size_t>(final_grid_size));
  SearchInRegion(local_points, params.func, current_region, final_grid_size, 1);

  Point best_point;
  best_point = local_points[0];
  if (params.use_local_optimization) {
    best_point = LocalOptimization(params.func, best_point.x, best_point.y, params.x_min, params.x_max, params.y_min,
                                   params.y_max);
  }

  result.x_min = best_point.x;
  result.y_min = best_point.y;
  result.value = best_point.value;

  return true;
}

bool LevonychevIMultistep2dOptimizationSEQ::PostProcessingImpl() {
  const auto &params = GetInput();
  auto &result = GetOutput();

  return result.x_min >= params.x_min && result.x_min <= params.x_max && result.y_min >= params.y_min &&
         result.y_min <= params.y_max;
}

}  // namespace levonychev_i_multistep_2d_optimization
