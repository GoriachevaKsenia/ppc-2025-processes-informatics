#include "khruev_a_global_opt/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <functional>
#include <utility>
#include <vector>

#include "khruev_a_global_opt/common/include/common.hpp"

namespace khruev_a_global_opt {

KhruevAGlobalOptMPI::KhruevAGlobalOptMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

bool KhruevAGlobalOptMPI::ValidationImpl() {
  return GetInput().max_iter > 0 && GetInput().epsilon > 0 && GetInput().r > 1.0;
}

bool KhruevAGlobalOptMPI::PreProcessingImpl() {
  trials_.clear();
  return true;
}

double KhruevAGlobalOptMPI::CalculateFunction(double t) {
  double u = 0;
  double v = 0;
  D2xy(t, u, v);
  double real_x = GetInput().ax + (u * (GetInput().bx - GetInput().ax));
  double real_y = GetInput().ay + (v * (GetInput().by - GetInput().ay));
  return TargetFunction(GetInput().func_id, real_x, real_y);
}

void KhruevAGlobalOptMPI::AddTrialUnsorted(double t, double z) {
  Trial tr{};
  tr.x = t;
  tr.z = z;
  trials_.push_back(tr);
}

double KhruevAGlobalOptMPI::ComputeM() {
  double max_slope = 0.0;
  for (size_t i = 1; i < trials_.size(); ++i) {
    double dx = trials_[i].x - trials_[i - 1].x;
    double dz = std::abs(trials_[i].z - trials_[i - 1].z);
    if (dx > 1e-12) {
      max_slope = std::max(max_slope, dz / dx);
    }
  }
  return (max_slope > 0) ? GetInput().r * max_slope : 1.0;
}

std::vector<KhruevAGlobalOptMPI::IntervalInfo> KhruevAGlobalOptMPI::ComputeIntervals(double m) const {
  std::vector<IntervalInfo> intervals;
  intervals.reserve(trials_.size() - 1);
  for (size_t i = 1; i < trials_.size(); ++i) {
    double dx = trials_[i].x - trials_[i - 1].x;
    double z_r = trials_[i].z;
    double z_l = trials_[i - 1].z;

    // Классическая формула Стронгина
    double r = (m * dx) + (((z_r - z_l) * (z_r - z_l)) / (m * dx)) - (2.0 * (z_r + z_l));
    intervals.push_back({r, static_cast<int>(i)});
  }
  return intervals;
}

bool KhruevAGlobalOptMPI::LocalShouldStop(const std::vector<IntervalInfo> &intervals, int num_to_check) {
  bool local_stop = true;
  for (int i = 0; i < num_to_check; ++i) {
    int idx = intervals[i].index;
    if (trials_[idx].x - trials_[idx - 1].x > GetInput().epsilon) {
      local_stop = false;
      break;
    }
  }
  return local_stop;
}

double KhruevAGlobalOptMPI::GenerateNewX(int idx, double m) const {
  double x_r = trials_[idx].x;
  double x_l = trials_[idx - 1].x;
  double z_r = trials_[idx].z;
  double z_l = trials_[idx - 1].z;

  // Формула вычисления новой точки испытания
  double new_x = (0.5 * (x_r + x_l)) - ((z_r - z_l) / (2.0 * m));

  // Защита от выхода за границы интервала
  if (new_x <= x_l || new_x >= x_r) {
    new_x = 0.5 * (x_r + x_l);
  }
  return new_x;
}

void KhruevAGlobalOptMPI::CollectAndAddPoints(const std::vector<Point> &global_res, int &k) {
  for (auto global_re : global_res) {
    if (global_re.x >= 0.0) {
      AddTrialUnsorted(global_re.x, global_re.z);
      k++;
    }
  }
}

bool KhruevAGlobalOptMPI::RunImpl() {
  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  // Начальные испытания
  if (trials_.empty()) {
    AddTrialUnsorted(0.0, CalculateFunction(0.0));
    AddTrialUnsorted(1.0, CalculateFunction(1.0));
  }
  std::ranges::sort(trials_, [](const Trial &a, const Trial &b) { return a.x < b.x; });

  int k = 2;  // Уже провели 2 испытания

  while (k < GetInput().max_iter) {
    // 1. Вычисляем M (одинаково на всех процессах)
    double m = ComputeM();

    // 2. Вычисляем характеристики R
    auto intervals = ComputeIntervals(m);

    // 3. Сортируем интервалы по убыванию характеристики
    std::ranges::sort(intervals, std::greater<>());

    // 4. Проверка критерия остановки (по самому широкому из лучших интервалов)
    int num_to_check = std::min(static_cast<int>(intervals.size()), size);
    bool local_stop = LocalShouldStop(intervals, num_to_check);

    // Синхронная остановка
    int stop_signal = 0;
    int local_stop_signal = local_stop ? 1 : 0;
    MPI_Allreduce(&local_stop_signal, &stop_signal, 1, MPI_INT, MPI_LAND, MPI_COMM_WORLD);
    if (stop_signal != 0) {
      break;
    }

    // 5. Генерация новых точек
    Point my_point{.x = -1.0, .z = 0.0};

    if (std::cmp_less(rank, static_cast<int>(intervals.size()))) {
      int idx = intervals[rank].index;
      double new_x = GenerateNewX(idx, m);

      my_point.x = new_x;
      my_point.z = CalculateFunction(new_x);
    }

    // 6. Сбор результатов со всех процессов
    std::vector<Point> global_res(size);
    MPI_Allgather(&my_point, 2, MPI_DOUBLE, global_res.data(), 2, MPI_DOUBLE, MPI_COMM_WORLD);

    // 7. Обновление списка
    CollectAndAddPoints(global_res, k);
    std::ranges::sort(trials_, [](const Trial &a, const Trial &b) { return a.x < b.x; });
  }

  // Поиск финального минимума среди всех испытаний
  auto it = std::ranges::min_element(trials_, {}, &Trial::z);

  double u = 0;
  double v = 0;
  D2xy(it->x, u, v);
  result_.x = GetInput().ax + (u * (GetInput().bx - GetInput().ax));
  result_.y = GetInput().ay + (v * (GetInput().by - GetInput().ay));
  result_.value = it->z;
  result_.iter_count = k;

  GetOutput() = result_;
  return true;
}

bool KhruevAGlobalOptMPI::PostProcessingImpl() {
  return true;
}

}  // namespace khruev_a_global_opt
