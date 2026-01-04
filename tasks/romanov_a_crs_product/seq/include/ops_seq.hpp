#pragma once

#include "romanov_a_crs_product/common/include/common.hpp"
#include "task/include/task.hpp"

namespace romanov_a_crs_product {

class RomanovACRSProductSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }
  explicit RomanovACRSProductSEQ(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace romanov_a_crs_product
