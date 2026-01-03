#pragma once

#include "rozenberg_a_radix_simple_merge/common/include/common.hpp"
#include "task/include/task.hpp"

namespace rozenberg_a_radix_simple_merge {

class RozenbergARadixSimpleMergeSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }
  explicit RozenbergARadixSimpleMergeSEQ(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace rozenberg_a_radix_simple_merge
