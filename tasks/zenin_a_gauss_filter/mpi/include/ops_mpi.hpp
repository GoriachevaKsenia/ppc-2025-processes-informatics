#pragma once

#include "task/include/task.hpp"
#include "zenin_a_gauss_filter/common/include/common.hpp"

namespace zenin_a_gauss_filter {

class ZeninAGaussFilterMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit ZeninAGaussFilterMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
  int proc_num_ = 1;
  int width_ = 0;
  int height_ = 0;
  int channels_ = 0;

  int grid_rows_ = 1;
  int grid_cols_ = 1;

  int block_h_ = 0;
  int block_w_ = 0;
  int extra_h_ = 0;
  int extra_w_ = 0;
};

}  // namespace zenin_a_gauss_filter
