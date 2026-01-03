#pragma once

#include <string>
#include <vector>

#include "akimov_i_star/common/include/common.hpp"
#include "task/include/task.hpp"

namespace akimov_i_star {

class AkimovIStarSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }
  explicit AkimovIStarSEQ(const InType &in);

  struct Op {
    int src = 0;
    int dst = 0;
    std::string msg;
  };

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  InType input_buffer_;
  int received_count_ = 0;
  std::vector<Op> ops_;
};

}  // namespace akimov_i_star
