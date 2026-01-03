#pragma once

#include <tuple>
#include <vector>

#include "chaschin_v_sobel_operator/common/include/common.hpp"
#include "task/include/task.hpp"

namespace chaschin_v_sobel_operator {

class ChaschinVSobelOperatorMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit ChaschinVSobelOperatorMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  static std::vector<float> PreprocessToGrayscaleWithOverlap(const std::vector<std::vector<Pixel>> &image, int n_procs,
                                                             std::vector<int> &sendcounts, std::vector<int> &displs);

  static float SobelAt(const std::vector<float> &img, int i, int j, int stride);
  std::vector<float> PreProcessGray_;
  std::vector<float> PostProcessGray_;
  std::tuple<int, int> Size_;
  std::vector<int> ScatterSendCounts_;
  std::vector<int> ScatterDispls_;
};

}  // namespace chaschin_v_sobel_operator
