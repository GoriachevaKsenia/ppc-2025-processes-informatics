#pragma once

#include <cstdint>

#include "mityaeva_d_contrast_enhancement_histogram_stretching/common/include/common.hpp"
#include "task/include/task.hpp"

namespace mityaeva_d_contrast_enhancement_histogram_stretching {

class ContrastEnhancementSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }

  explicit ContrastEnhancementSEQ(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  int width_{0};
  int height_{0};
  int total_pixels_{0};
  uint8_t min_pixel_{255};
  uint8_t max_pixel_{0};
};

}  // namespace mityaeva_d_contrast_enhancement_histogram_stretching
