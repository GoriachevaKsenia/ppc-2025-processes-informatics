#pragma once

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <vector>

namespace levonychev_i_multistep_2d_optimization {

struct Point {
  double x = 0.0;
  double y = 0.0;
  double value = std::numeric_limits<double>::max();

  Point() = default;
  Point(double x_val, double y_val, double func_value) : x(x_val), y(y_val), value(func_value) {}
};

struct SearchRegion {
  double x_min = 0.0;
  double x_max = 0.0;
  double y_min = 0.0;
  double y_max = 0.0;

  SearchRegion() = default;
  SearchRegion(double x_mn, double x_mx, double y_mn, double y_mx)
      : x_min(x_mn), x_max(x_mx), y_min(y_mn), y_max(y_mx) {}
};

inline Point LocalOptimization(const std::function<double(double, double)> &func, double x_start, double y_start,
                               double x_min, double x_max, double y_min, double y_max, int max_iterations = 100) {
  const double epsilon = 1e-6;
  const double step_size = 0.01;

  double x = x_start;
  double y = y_start;
  double current_value = func(x, y);

  for (int iter = 0; iter < max_iterations; ++iter) {
    const double h = 1e-5;
    double grad_x = (func(x + h, y) - func(x - h, y)) / (2.0 * h);
    double grad_y = (func(x, y + h) - func(x, y - h)) / (2.0 * h);

    double new_x = x - (step_size * grad_x);
    double new_y = y - (step_size * grad_y);

    new_x = std::max(x_min, std::min(x_max, new_x));
    new_y = std::max(y_min, std::min(y_max, new_y));

    double new_value = func(new_x, new_y);

    if (std::abs(new_value - current_value) < epsilon) {
      break;
    }

    x = new_x;
    y = new_y;
    current_value = new_value;
  }

  return {x, y, current_value};
}

inline void SearchInRegion(std::vector<Point> &grid_points, const std::function<double(double, double)> &func,
                           const SearchRegion &region, int grid_size, int size) {
  auto grid_size_double = static_cast<double>(grid_size);
  auto size_double = static_cast<double>(size);

  double step_x = (region.x_max - region.x_min) / (grid_size_double / size_double - 1.0);
  double step_y = (region.y_max - region.y_min) / (grid_size_double - 1.0);

  for (int i = 0; i < grid_size / size; ++i) {
    for (int j = 0; j < grid_size; ++j) {
      double x = region.x_min + (i * step_x);
      double y = region.y_min + (j * step_y);

      double value = func(x, y);
      grid_points.emplace_back(x, y, value);
    }
  }

  std::ranges::sort(grid_points, [](const Point &a, const Point &b) { return a.value < b.value; });
}

}  // namespace levonychev_i_multistep_2d_optimization
