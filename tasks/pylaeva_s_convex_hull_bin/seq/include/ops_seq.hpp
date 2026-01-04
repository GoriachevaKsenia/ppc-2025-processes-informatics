#pragma once

#include <vector>

#include "pylaeva_s_convex_hull_bin/common/include/common.hpp"
#include "task/include/task.hpp"

namespace pylaeva_s_convex_hull_bin {

class PylaevaSConvexHullBinSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }
  explicit PylaevaSConvexHullBinSEQ(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  void FindConnectedComponents();
  static std::vector<Point> GrahamScan(const std::vector<Point> &points);

  ImageData processed_data_;
};

}  // namespace pylaeva_s_convex_hull_bin
