#pragma once

#include <vector>

#include "chaschin_v_sobel_operator/common/include/common.hpp"
#include "task/include/task.hpp"

namespace chaschin_v_sobel_operator {

class ChaschinVSobelOperatorSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }
  explicit ChaschinVSobelOperatorSEQ(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

std::vector<float> SobelSeq(const std::vector<std::vector<float>> &image);

}  // namespace chaschin_v_sobel_operator
