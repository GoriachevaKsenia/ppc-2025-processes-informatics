#pragma once

#include <cstdint>
#include <queue>
#include <vector>

#include "pylaeva_s_convex_hull_bin/common/include/common.hpp"
#include "task/include/task.hpp"

namespace pylaeva_s_convex_hull_bin {

class PylaevaSConvexHullBinMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit PylaevaSConvexHullBinMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  void FindConnectedComponentsMpi();
  void ProcessComponentsAndComputeHulls();
  void GatherConvexHullsToRank0();
  void ReceiveHullsFromProcess(int source_rank, int hull_count);
  void SendHullsToRank0();
  static std::vector<Point> GrahamScan(const std::vector<Point> &points);

  void ScatterDataAndDistributeWork();
  void ExchangeBoundaryRows(int width, int local_rows, int extended_start_row, int extended_local_rows,
                            std::vector<uint8_t> &extended_pixels) const;
  static void ProcessExtendedRegion(int width, int extended_start_row, int extended_local_rows,
                                    const std::vector<uint8_t> &extended_pixels, std::vector<bool> &visited_extended,
                                    std::vector<std::vector<Point>> &all_components);
  static void ProcessExtendedNeighbors(const Point &p, int width, int extended_start_row, int extended_local_rows,
                                       const std::vector<uint8_t> &extended_pixels, std::vector<bool> &visited_extended,
                                       std::queue<Point> &q);
  void FilterLocalComponents(const std::vector<std::vector<Point>> &all_components);

  ImageData local_data_;
  ImageData full_data_;
  int rank_{0}, size_{0};
  int start_row_{0}, end_row_{0};
  int rows_per_proc_{0};
};

}  // namespace pylaeva_s_convex_hull_bin
