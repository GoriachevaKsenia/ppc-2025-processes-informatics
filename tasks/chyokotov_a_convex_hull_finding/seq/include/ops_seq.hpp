#pragma once

#include <utility>
#include <vector>

#include "chyokotov_a_convex_hull_finding/common/include/common.hpp"
#include "task/include/task.hpp"

namespace chyokotov_a_convex_hull_finding {

class ChyokotovConvexHullFindingSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }
  explicit ChyokotovConvexHullFindingSEQ(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  static std::vector<std::pair<int, int>> Bfs(int start_x, int start_y, const std::vector<std::vector<int>> &picture,
                                              std::vector<std::vector<bool>> &visited);
  std::vector<std::vector<std::pair<int, int>>> FindComponent();
  static int Cross(const std::pair<int, int> &o, const std::pair<int, int> &a, const std::pair<int, int> &b);
  static std::vector<std::pair<int, int>> ConvexHull(std::vector<std::pair<int, int>> x);
};

}  // namespace chyokotov_a_convex_hull_finding
