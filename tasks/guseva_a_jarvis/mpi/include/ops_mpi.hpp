#pragma once

#include <utility>
#include <vector>

#include "guseva_a_jarvis/common/include/common.hpp"
#include "task/include/task.hpp"

namespace guseva_a_jarvis {

class GusevaAJarvisMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit GusevaAJarvisMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  [[nodiscard]] std::pair<int, int> FindStartPoint() const;
  [[nodiscard]] std::pair<int, int> FindNextPoint(const std::pair<int, int> &current_point,
                                                  const std::pair<int, int> &candidate_point) const;

  int rank_ = 0;
  int size_ = 1;

  std::vector<std::pair<int, int>> points_;
  std::vector<std::pair<int, int>> hull_;
};

}  // namespace guseva_a_jarvis
