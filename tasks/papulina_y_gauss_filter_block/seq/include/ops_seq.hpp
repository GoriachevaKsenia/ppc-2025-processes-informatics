#pragma once

#include <vector>

#include "papulina_y_gauss_filter_block/common/include/common.hpp"
#include "task/include/task.hpp"
namespace papulina_y_gauss_filter_block {

class PapulinaYGaussFilterSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }
  explicit PapulinaYGaussFilterSEQ(const InType &in);

 private:
  std::vector<unsigned char> NewPixel(const int &row, const int &col);
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
  Picture Pic_;
  void ClampCoordinates(int &n_x, int &n_y) const;
};

}  // namespace papulina_y_gauss_filter_block
