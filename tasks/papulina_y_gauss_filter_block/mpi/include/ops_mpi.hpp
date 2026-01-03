#pragma once

#include <vector>

#include "papulina_y_gauss_filter_block/common/include/common.hpp"
#include "task/include/task.hpp"
namespace papulina_y_gauss_filter_block {

struct Block {
  int my_block_rows = 0;
  int my_block_cols = 0;
  int expanded_rows = 0;
  int expanded_cols = 0;
  int start_row = 0;
  int start_col = 0;
  Block(int mb_rows, int mb_cols, int exp_rows, int exp_cols, int s_row, int s_col)
      : my_block_rows(mb_rows),
        my_block_cols(mb_cols),
        expanded_rows(exp_rows),
        expanded_cols(exp_cols),
        start_row(s_row),
        start_col(s_col) {}
};

class PapulinaYGaussFilterMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit PapulinaYGaussFilterMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
  Picture Pic_;
  int procNum_ = 0;
  int height_ = 0;
  int width_ = 0;
  int channels_ = 0;
  int overlap_ = 1;
  int grid_rows_ = 0;
  int grid_cols_ = 0;
  int block_rows_ = 0;
  int block_cols_ = 0;
  int extra_rows_ = 0;
  int extra_cols_ = 0;
  void CalculateBlock(const Block &block, std::vector<unsigned char> &my_block);
  void DataDistribution();
  void NewBlock(const Block &block, const std::vector<unsigned char> &my_block,
                std::vector<unsigned char> &filtered_block) const;
  void GetResult(const int &rank, const Block &block, const std::vector<unsigned char> &filtered_block);
  void ExtractBlock(const Block &block, const std::vector<unsigned char> &filtered_block,
                    std::vector<unsigned char> &my_result) const;
  void FillImage(const Block &block, const std::vector<unsigned char> &my_result,
                 std::vector<unsigned char> &final_image) const;
  static void ClampCoordinates(int &global_i, int &global_j, int height, int width);
};

}  // namespace papulina_y_gauss_filter_block
