#pragma once

#include <cstddef>
#include <vector>

#include "leonova_a_star/common/include/common.hpp"
#include "task/include/task.hpp"

namespace leonova_a_star {

class LeonovaAStarSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }
  explicit LeonovaAStarSEQ(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  bool ResizeOutputMatrix(size_t rows, size_t cols);
  void MultiplyMatrices(const std::vector<std::vector<int>> &matrix_a, const std::vector<std::vector<int>> &matrix_b,
                        size_t rows_a, size_t cols_a, size_t cols_b);
};

}  // namespace leonova_a_star
