#pragma once

#include <cstdint>
#include <vector>

#include "nikolaev_d_most_dif_vec_neighbors/common/include/common.hpp"
#include "task/include/task.hpp"

namespace nikolaev_d_most_dif_vec_neighbors {

class NikolaevDMostDifVecNeighborsMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit NikolaevDMostDifVecNeighborsMPI(const InType &in);

 private:
  struct LocalMaxInfo {
    int64_t diff = -1;
    int pair_first = 0;
    int pair_second = 0;
    int first_elem = 0;
    int last_elem = 0;
  };

  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  void FindLocalDiff(int rank, int size, int actual_processes, std::vector<int> &local_data, int local_size);
  void ProcessLocalData(int rank, int actual_processes, std::vector<LocalMaxInfo> &all_info);
};

}  // namespace nikolaev_d_most_dif_vec_neighbors
