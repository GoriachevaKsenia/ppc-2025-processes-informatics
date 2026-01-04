#pragma once

#include "gasenin_l_image_smooth/common/include/common.hpp"
#include "task/include/task.hpp"

namespace gasenin_l_image_smooth {

class GaseninLImageSmoothMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit GaseninLImageSmoothMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace gasenin_l_image_smooth
