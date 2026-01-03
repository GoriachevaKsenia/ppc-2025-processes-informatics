#pragma once

#include <string>
#include <vector>

#include "akimov_i_star/common/include/common.hpp"
#include "task/include/task.hpp"

namespace akimov_i_star {

class AkimovIStarMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit AkimovIStarMPI(const InType &in);

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
  std::vector<Op> ops_;
  int received_count_ = 0;
};
}  // namespace akimov_i_star
