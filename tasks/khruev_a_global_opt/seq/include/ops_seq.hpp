#pragma once

#include <vector>

#include "khruev_a_global_opt/common/include/common.hpp"
#include "task/include/task.hpp"

namespace khruev_a_global_opt {

class KhruevAGlobalOptSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }

  explicit KhruevAGlobalOptSEQ(const InType &in);
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  std::vector<Trial> trials_;
  OutType result_{};

  double CalculateFunction(double t);
  void AddTrial(double t, double z);

  double ComputeM();
  [[nodiscard]] int FindBestInterval(double m) const;
  [[nodiscard]] double GenerateNewX(int best_interval, double m) const;
};

}  // namespace khruev_a_global_opt
