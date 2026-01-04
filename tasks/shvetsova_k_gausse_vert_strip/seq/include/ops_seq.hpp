#pragma once

#include <vector>

#include "shvetsova_k_gausse_vert_strip/common/include/common.hpp"
#include "task/include/task.hpp"

namespace shvetsova_k_gausse_vert_strip {

class ShvetsovaKGaussVertStripSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }
  explicit ShvetsovaKGaussVertStripSEQ(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  int size_of_rib_ = 0;
  void ForwardElimination(int n, std::vector<std::vector<double>> &a, std::vector<double> &x) const;
  [[nodiscard]] std::vector<double> BackSubstitution(int n, const std::vector<std::vector<double>> &a,
                                                     const std::vector<double> &x) const;
  static void ProcessRow(int i, int row_end, std::vector<std::vector<double>> &a, std::vector<double> &x, double eps);
};

}  // namespace shvetsova_k_gausse_vert_strip
