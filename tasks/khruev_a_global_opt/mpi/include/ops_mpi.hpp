#pragma once

#include <vector>

#include "khruev_a_global_opt/common/include/common.hpp"
#include "task/include/task.hpp"

namespace khruev_a_global_opt {

class KhruevAGlobalOptMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }

  explicit KhruevAGlobalOptMPI(const InType &in);
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  struct IntervalInfo {
    double R;
    int index;  // индекс в массиве trials_ (указывает на правый конец интервала)

    bool operator>(const IntervalInfo &other) const {
      return R > other.R;  // Для сортировки по убыванию
    }
  };

  struct Point {
    double x, z;
  };

  std::vector<Trial> trials_;
  OutType result_{};

  double CalculateFunction(double t);
  void AddTrialUnsorted(double t, double z);

  double ComputeM();
  [[nodiscard]] std::vector<IntervalInfo> ComputeIntervals(double m) const;
  bool LocalShouldStop(const std::vector<IntervalInfo> &intervals, int num_to_check);
  [[nodiscard]] double GenerateNewX(int idx, double m) const;
  void CollectAndAddPoints(const std::vector<Point> &global_res, int &k);
};

}  // namespace khruev_a_global_opt
