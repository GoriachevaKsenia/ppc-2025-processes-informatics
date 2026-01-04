#include "khruev_a_global_opt/seq/include/ops_seq.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>

#include "khruev_a_global_opt/common/include/common.hpp"

namespace khruev_a_global_opt {

KhruevAGlobalOptSEQ::KhruevAGlobalOptSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

bool KhruevAGlobalOptSEQ::ValidationImpl() {
  return GetInput().max_iter > 0 && GetInput().epsilon > 0 && GetInput().r > 1.0;
}

bool KhruevAGlobalOptSEQ::PreProcessingImpl() {
  trials_.clear();
  return true;
}

double KhruevAGlobalOptSEQ::CalculateFunction(double t) {
  double u = 0.0;
  double v = 0.0;
  D2xy(t, u, v);  // Преобразование Гильберта 1D -> 2D

  // Масштабирование из [0,1] в реальную область [ax, bx]
  double real_x = GetInput().ax + (u * (GetInput().bx - GetInput().ax));
  double real_y = GetInput().ay + (v * (GetInput().by - GetInput().ay));

  return TargetFunction(GetInput().func_id, real_x, real_y);
}

void KhruevAGlobalOptSEQ::AddTrial(double t, double z) {
  Trial tr{};
  tr.x = t;
  tr.z = z;
  trials_.push_back(tr);
  // Сортировка по координате x (1)
  std::ranges::sort(trials_, [](const Trial &a, const Trial &b) { return a.x < b.x; });
}

double KhruevAGlobalOptSEQ::ComputeM() {
  double max_slope = 0.0;
  for (size_t i = 1; i < trials_.size(); ++i) {
    double dx = trials_[i].x - trials_[i - 1].x;
    double dz = std::abs(trials_[i].z - trials_[i - 1].z);
    if (dx > 1e-15) {
      max_slope = std::max(max_slope, dz / dx);
    }
  }
  return (max_slope > 0) ? GetInput().r * max_slope : 1.0;
}

int KhruevAGlobalOptSEQ::FindBestInterval(double m) const {
  double max_r = -1e18;
  int best_interval = -1;

  for (size_t i = 1; i < trials_.size(); ++i) {
    double dx = trials_[i].x - trials_[i - 1].x;
    double z_r = trials_[i].z;
    double z_l = trials_[i - 1].z;

    // формула (5)
    double r = (m * dx) + (((z_r - z_l) * (z_r - z_l)) / (m * dx)) - (2.0 * (z_r + z_l));

    if (r > max_r) {
      max_r = r;
      best_interval = static_cast<int>(i);
    }
  }
  return best_interval;
}

double KhruevAGlobalOptSEQ::GenerateNewX(int best_interval, double m) const {
  double x_r = trials_[best_interval].x;
  double x_l = trials_[best_interval - 1].x;
  double z_r = trials_[best_interval].z;
  double z_l = trials_[best_interval - 1].z;

  double new_x = (0.5 * (x_r + x_l)) - ((z_r - z_l) / (2.0 * m));

  if (new_x < x_l) {
    new_x = x_l + 1e-9;
  }
  if (new_x > x_r) {
    new_x = x_r - 1e-9;
  }
  return new_x;
}

bool KhruevAGlobalOptSEQ::RunImpl() {
  AddTrial(0.0, CalculateFunction(0.0));
  AddTrial(1.0, CalculateFunction(1.0));

  int k = 1;

  for (k = 1; k < GetInput().max_iter; ++k) {
    double m = ComputeM();

    int best_interval = FindBestInterval(m);

    if (best_interval == -1) {
      break;
    }

    double dx_best = trials_[best_interval].x - trials_[best_interval - 1].x;
    if (dx_best < GetInput().epsilon) {
      break;
    }

    double new_x = GenerateNewX(best_interval, m);

    AddTrial(new_x, CalculateFunction(new_x));
  }

  // Поиск минимума среди всех точек
  double min_z = 1e18;
  double best_t = 0;
  for (const auto &t : trials_) {
    if (t.z < min_z) {
      min_z = t.z;
      best_t = t.x;
    }
  }

  // Восстанавливаем 2D координаты для ответа
  double u = 0.0;
  double v = 0.0;
  D2xy(best_t, u, v);
  result_.x = GetInput().ax + (u * (GetInput().bx - GetInput().ax));
  result_.y = GetInput().ay + (v * (GetInput().by - GetInput().ay));
  result_.value = min_z;
  result_.iter_count = k;

  GetOutput() = result_;
  return true;
}

bool KhruevAGlobalOptSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace khruev_a_global_opt
