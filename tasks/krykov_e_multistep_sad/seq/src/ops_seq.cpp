#include "krykov_e_multistep_sad/seq/include/ops_seq.hpp"

#include <algorithm>
#include <cmath>
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

Region SplitRegionX(const Region &r, double xm) {
  return Region{.x_min = r.x_min, .x_max = xm, .y_min = r.y_min, .y_max = r.y_max, .value = 0.0};
}

Region SplitRegionXRight(const Region &r, double xm) {
  return Region{.x_min = xm, .x_max = r.x_max, .y_min = r.y_min, .y_max = r.y_max, .value = 0.0};
}

Region SplitRegionY(const Region &r, double ym) {
  return Region{.x_min = r.x_min, .x_max = r.x_max, .y_min = r.y_min, .y_max = ym, .value = 0.0};
}

Region SplitRegionYTop(const Region &r, double ym) {
  return Region{.x_min = r.x_min, .x_max = r.x_max, .y_min = ym, .y_max = r.y_max, .value = 0.0};
}

}  // namespace

KrykovEMultistepSADSEQ::KrykovEMultistepSADSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

bool KrykovEMultistepSADSEQ::ValidationImpl() {
  const auto &[f, x1, x2, y1, y2] = GetInput();
  return static_cast<bool>(f) && x1 < x2 && y1 < y2;
}

bool KrykovEMultistepSADSEQ::PreProcessingImpl() {
  return true;
}

bool KrykovEMultistepSADSEQ::RunImpl() {
  const auto &[f, x_min, x_max, y_min, y_max] = GetInput();

  std::vector<Region> regions;
  regions.push_back(Region{.x_min = x_min, .x_max = x_max, .y_min = y_min, .y_max = y_max, .value = 0.0});
  EvaluateCenter(f, regions.front());

  for (int iter = 0; iter < kMaxIter; ++iter) {
    for (auto &r : regions) {
      EvaluateCenter(f, r);
    }

    auto best_it = std::ranges::min_element(regions, {}, &Region::value);
    Region best = *best_it;
    regions.erase(best_it);

    double dx = best.x_max - best.x_min;
    double dy = best.y_max - best.y_min;

    if (std::max(dx, dy) < kEps) {
      regions.push_back(best);
      break;
    }

    if (dx >= dy) {
      double xm = 0.5 * (best.x_min + best.x_max);
      Region r1 = SplitRegionX(best, xm);
      Region r2 = SplitRegionXRight(best, xm);
      EvaluateCenter(f, r1);
      EvaluateCenter(f, r2);
      regions.push_back(r1);
      regions.push_back(r2);
    } else {
      double ym = 0.5 * (best.y_min + best.y_max);
      Region r1 = SplitRegionY(best, ym);
      Region r2 = SplitRegionYTop(best, ym);
      EvaluateCenter(f, r1);
      EvaluateCenter(f, r2);
      regions.push_back(r1);
      regions.push_back(r2);
    }
  }

  const auto &best = *std::ranges::min_element(regions, {}, &Region::value);
  double x = 0.5 * (best.x_min + best.x_max);
  double y = 0.5 * (best.y_min + best.y_max);
  GetOutput() = {x, y, best.value};

  return true;
}

bool KrykovEMultistepSADSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace krykov_e_multistep_sad
