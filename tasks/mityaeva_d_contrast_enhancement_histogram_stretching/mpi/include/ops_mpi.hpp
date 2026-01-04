#pragma once

#include <cstdint>
#include <vector>

#include "mityaeva_d_contrast_enhancement_histogram_stretching/common/include/common.hpp"
#include "task/include/task.hpp"

namespace mityaeva_d_contrast_enhancement_histogram_stretching {

class ContrastEnhancementMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }

  explicit ContrastEnhancementMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  void CalculateDistribution(int rank, int size, int &my_pixels, int &my_offset) const;
  void GatherResults(int rank, int size, const std::vector<uint8_t> &local_result,
                     std::vector<uint8_t> &final_output) const;

  int width_{0};
  int height_{0};
  int total_pixels_{0};
};

}  // namespace mityaeva_d_contrast_enhancement_histogram_stretching
