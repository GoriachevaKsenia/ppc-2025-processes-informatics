#include "kruglova_a_2d_multistep_par_opt/seq/include/ops_seq.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <functional>
#include <limits>
#include <utility>
#include <vector>

#include "kruglova_a_2d_multistep_par_opt/common/include/common.hpp"

namespace kruglova_a_2d_multistep_par_opt {

namespace {

struct Trial1D {
  double x;
  double z;

  Trial1D(double x_val, double z_val) : x(x_val), z(z_val) {}
};

struct Trial2D {
  double x;
  double y;
  double z;

  Trial2D(double x_val, double y_val, double z_val) : x(x_val), y(y_val), z(z_val) {}
};

template <typename T>
double CalculateM(const std::vector<T> &trials) {
  double m_max = 0.0;
  for (size_t i = 0; i + 1 < trials.size(); ++i) {
    const double dx = trials[i + 1].x - trials[i].x;
    if (dx > 1e-15) {
      const double dz = std::abs(trials[i + 1].z - trials[i].z);
      const double ratio = dz / dx;
      m_max = std::max(ratio, m_max);
    }
  }
  return m_max;
}

size_t FindBestInterval1D(const std::vector<Trial1D> &trials, double m_scaled) {
  double max_rate = -std::numeric_limits<double>::infinity();
  size_t best_idx = 0;

  for (size_t i = 0; i + 1 < trials.size(); ++i) {
    const double dx = trials[i + 1].x - trials[i].x;
    const double dz = trials[i + 1].z - trials[i].z;

    const double term1 = m_scaled * dx;
    const double term2 = (dz * dz) / (m_scaled * dx);
    const double term3 = 2.0 * (trials[i + 1].z + trials[i].z);
    const double rate = term1 + term2 - term3;

    if (rate > max_rate) {
      max_rate = rate;
      best_idx = i;
    }
  }

  return best_idx;
}

size_t FindBestInterval2D(const std::vector<Trial2D> &trials, double m_scaled) {
  double max_rate = -std::numeric_limits<double>::infinity();
  size_t best_idx = 0;

  for (size_t i = 0; i + 1 < trials.size(); ++i) {
    const double dx = trials[i + 1].x - trials[i].x;
    const double dz = trials[i + 1].z - trials[i].z;

    const double term1 = m_scaled * dx;
    const double term2 = (dz * dz) / (m_scaled * dx);
    const double term3 = 2.0 * (trials[i + 1].z + trials[i].z);
    const double rate = term1 + term2 - term3;

    if (rate > max_rate) {
      max_rate = rate;
      best_idx = i;
    }
  }

  return best_idx;
}

size_t FindBestZ1D(const std::vector<Trial1D> &trials) {
  size_t best = 0;
  for (size_t i = 1; i < trials.size(); ++i) {
    if (trials[i].z < trials[best].z) {
      best = i;
    }
  }
  return best;
}

size_t FindBestZ2D(const std::vector<Trial2D> &trials) {
  size_t best = 0;
  for (size_t i = 1; i < trials.size(); ++i) {
    if (trials[i].z < trials[best].z) {
      best = i;
    }
  }
  return best;
}

void InsertSorted1D(std::vector<Trial1D> &trials, const Trial1D &value) {
  size_t pos = 0;
  while (pos < trials.size() && trials[pos].x < value.x) {
    ++pos;
  }
  trials.insert(trials.begin() + static_cast<std::ptrdiff_t>(pos), value);
}

void InsertSorted2D(std::vector<Trial2D> &trials, const Trial2D &value) {
  size_t pos = 0;
  while (pos < trials.size() && trials[pos].x < value.x) {
    ++pos;
  }

  if (pos == trials.size() || std::abs(trials[pos].x - value.x) > 1e-12) {
    trials.insert(trials.begin() + static_cast<std::ptrdiff_t>(pos), value);
  }
}

double Solve1DStrongin(const std::function<double(double)> &func, double a, double b, double eps, int max_iters,
                       double &best_x) {
  const double r_param = 2.0;

  std::vector<Trial1D> trials;
  trials.emplace_back(a, func(a));
  trials.emplace_back(b, func(b));

  if (trials[0].x > trials[1].x) {
    std::swap(trials[0], trials[1]);
  }

  for (int iter = 0; iter < max_iters; ++iter) {
    const double m_val = CalculateM(trials);
    const double m_scaled = (m_val > 0.0) ? (r_param * m_val) : 1.0;

    const size_t idx = FindBestInterval1D(trials, m_scaled);

    const double dx = trials[idx + 1].x - trials[idx].x;
    if (dx < eps) {
      break;
    }

    const double mid = 0.5 * (trials[idx + 1].x + trials[idx].x);
    const double diff = (trials[idx + 1].z - trials[idx].z) / (2.0 * m_scaled);
    const double x_new = mid - diff;

    InsertSorted1D(trials, Trial1D(x_new, func(x_new)));
  }

  const size_t best = FindBestZ1D(trials);
  best_x = trials[best].x;
  return trials[best].z;
}

}  // namespace

KruglovaA2DMuitSEQ::KruglovaA2DMuitSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

bool KruglovaA2DMuitSEQ::ValidationImpl() {
  const auto &in = GetInput();
  return in.x_max > in.x_min && in.y_max > in.y_min && in.eps > 0.0 && in.max_iters > 0;
}

bool KruglovaA2DMuitSEQ::PreProcessingImpl() {
  GetOutput() = {0.0, 0.0, std::numeric_limits<double>::max()};
  return true;
}

bool KruglovaA2DMuitSEQ::RunImpl() {
  const auto &in = GetInput();
  const double r_param = 2.0;

  auto compute_z = [&](double x_val, double &best_y) {
    return Solve1DStrongin([&](double y) { return ObjectiveFunction(x_val, y); }, in.y_min, in.y_max, in.eps,
                           std::max(50, in.max_iters / 10), best_y);
  };

  std::vector<Trial2D> x_trials;
  const int init_points = 20;

  for (int i = 0; i < init_points; ++i) {
    const double t = static_cast<double>(i) / static_cast<double>(init_points - 1);
    const double x = in.x_min + ((in.x_max - in.x_min) * t);

    double y = 0.0;
    double z = compute_z(x, y);
    x_trials.emplace_back(x, y, z);
  }

  for (size_t i = 0; i < x_trials.size(); ++i) {
    size_t min_idx = i;
    for (size_t j = i + 1; j < x_trials.size(); ++j) {
      if (x_trials[j].x < x_trials[min_idx].x) {
        min_idx = j;
      }
    }
    std::swap(x_trials[i], x_trials[min_idx]);
  }

  for (int iter = 0; iter < in.max_iters; ++iter) {
    const double m_val = CalculateM(x_trials);
    const double m_scaled = (m_val > 0.0) ? (r_param * m_val) : 1.0;

    const size_t idx = FindBestInterval2D(x_trials, m_scaled);
    const double dx = x_trials[idx + 1].x - x_trials[idx].x;

    if (dx < in.eps) {
      break;
    }

    const double mid = 0.5 * (x_trials[idx + 1].x + x_trials[idx].x);
    const double diff = (x_trials[idx + 1].z - x_trials[idx].z) / (2.0 * m_scaled);
    const double x_new = mid - diff;

    double y_new = 0.0;
    double z_new = compute_z(x_new, y_new);

    InsertSorted2D(x_trials, Trial2D(x_new, y_new, z_new));
  }

  const size_t best = FindBestZ2D(x_trials);
  GetOutput() = {x_trials[best].x, x_trials[best].y, x_trials[best].z};
  return true;
}

bool KruglovaA2DMuitSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace kruglova_a_2d_multistep_par_opt
