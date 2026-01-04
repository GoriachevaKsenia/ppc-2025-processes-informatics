#pragma once

#include <cstddef>
#include <utility>
#include <vector>

#include "guseva_a_jarvis/common/include/common.hpp"
#include "task/include/task.hpp"

namespace guseva_a_jarvis {

class GusevaAJarvisSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }
  explicit GusevaAJarvisSEQ(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  std::vector<std::pair<int, int>> points_;
  std::vector<std::pair<int, int>> hull_;

  [[nodiscard]] size_t FindFirst() const;
};

}  // namespace guseva_a_jarvis
