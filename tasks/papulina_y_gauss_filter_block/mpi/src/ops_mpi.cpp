#include "papulina_y_gauss_filter_block/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <vector>

#include "papulina_y_gauss_filter_block/common/include/common.hpp"

namespace papulina_y_gauss_filter_block {

PapulinaYGaussFilterMPI::PapulinaYGaussFilterMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = Picture();
}

bool PapulinaYGaussFilterMPI::ValidationImpl() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  int flag = 1;
  if (rank == 0) {
    flag = static_cast<int>(!(GetInput().pixels.empty()));
  }
  MPI_Bcast(&flag, 1, MPI_INT, 0, MPI_COMM_WORLD);
  return flag > 0;
}

bool PapulinaYGaussFilterMPI::PreProcessingImpl() {
  int rank = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &procNum_);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank == 0) {
    Pic_ = GetInput();  // считываем картинку в нулевой процесс
    height_ = Pic_.height;
    width_ = Pic_.width;
    channels_ = Pic_.channels;
  }
  // отправляем height_ и width_
  MPI_Bcast(&height_, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&width_, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&channels_, 1, MPI_INT, 0, MPI_COMM_WORLD);
  // определяем размеры сетки процессов
  for (int k = static_cast<int>(std::sqrt(procNum_)); k > 0; k--) {
    if (procNum_ % k == 0) {  // ищем наибольший делитель
      grid_rows_ = k;
      grid_cols_ = procNum_ / k;
      break;
    }
  }
  if (grid_rows_ == 0) {
    grid_rows_ = 1;
    grid_cols_ = procNum_;
  }
  // размер базового блока
  block_rows_ = height_ / grid_rows_;
  block_cols_ = width_ / grid_cols_;
  extra_rows_ = height_ % grid_rows_;
  extra_cols_ = width_ % grid_cols_;

  return true;
}

bool PapulinaYGaussFilterMPI::RunImpl() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  // координаты процесса в сетке
  int row = rank / grid_cols_;
  int col = rank % grid_cols_;

  // размеры блока конкретного процесса
  int my_block_rows = block_rows_ + (row < extra_rows_ ? 1 : 0);
  int my_block_cols = block_cols_ + (col < extra_cols_ ? 1 : 0);

  // рассширенный блок с учетом всех соседей
  int expanded_rows = my_block_rows + (2 * overlap_);
  int expanded_cols = my_block_cols + (2 * overlap_);

  // координаты начала блока конкретного процесса в сетке
  int start_row = (row * block_rows_) + std::min(row, extra_rows_);
  int start_col = (col * block_cols_) + std::min(col, extra_cols_);

  Block block(my_block_rows, my_block_cols, expanded_rows, expanded_cols, start_row, start_col);

  std::vector<unsigned char> my_block(static_cast<size_t>(expanded_rows * expanded_cols * channels_), 0);
  if (rank == 0) {
    CalculateBlock(block, my_block);
    DataDistribution();
  } else {
    MPI_Recv(my_block.data(), static_cast<int>(my_block.size()), MPI_UNSIGNED_CHAR, 0, 0, MPI_COMM_WORLD,
             MPI_STATUS_IGNORE);
  }

  std::vector<unsigned char> filtered_block(static_cast<size_t>(expanded_rows * expanded_cols * channels_), 0);
  NewBlock(block, my_block, filtered_block);
  GetResult(rank, block, filtered_block);

  return true;
}
void PapulinaYGaussFilterMPI::DataDistribution() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  for (int pr = 1; pr < procNum_; pr++) {
    int p_row = pr / grid_cols_;
    int p_col = pr % grid_cols_;

    int p_block_rows = block_rows_ + (p_row < extra_rows_ ? 1 : 0);
    int p_block_cols = block_cols_ + (p_col < extra_cols_ ? 1 : 0);

    int p_start_row = (p_row * block_rows_) + std::min(p_row, extra_rows_);
    int p_start_col = (p_col * block_cols_) + std::min(p_col, extra_cols_);

    int p_expanded_rows = p_block_rows + (2 * overlap_);
    int p_expanded_cols = p_block_cols + (2 * overlap_);
    Block block(p_block_rows, p_block_cols, p_expanded_rows, p_expanded_cols, p_start_row, p_start_col);
    std::vector<unsigned char> p_block(static_cast<size_t>(p_expanded_rows * p_expanded_cols * channels_), 0);

    CalculateBlock(block, p_block);

    MPI_Send(p_block.data(), static_cast<int>(p_block.size()), MPI_UNSIGNED_CHAR, pr, 0, MPI_COMM_WORLD);
  }
}
void PapulinaYGaussFilterMPI::ClampCoordinates(int &global_i, int &global_j, int height, int width) {
  global_i = std::max(0, std::min(height - 1, global_i));
  global_j = std::max(0, std::min(width - 1, global_j));
}
void PapulinaYGaussFilterMPI::CalculateBlock(const Block &block, std::vector<unsigned char> &my_block) {
  for (int i = -overlap_; i < block.my_block_rows + overlap_; i++) {
    for (int j = -overlap_; j < block.my_block_cols + overlap_; j++) {
      int global_i = block.start_row + i;
      int global_j = block.start_col + j;

      ClampCoordinates(global_i, global_j, height_, width_);

      for (int ch = 0; ch < channels_; ch++) {
        int local_idx = (((i + overlap_) * block.expanded_cols + (j + overlap_)) * channels_) + ch;
        int global_idx = ((global_i * width_ + global_j) * channels_) + ch;
        my_block[local_idx] = Pic_.pixels[global_idx];
      }
    }
  }
}

void PapulinaYGaussFilterMPI::NewBlock(const Block &block, const std::vector<unsigned char> &my_block,
                                       std::vector<unsigned char> &filtered_block) const {
  static constexpr std::array<float, 9> kErnel = {1.0F / 16, 2.0F / 16, 1.0F / 16, 2.0F / 16, 4.0F / 16,
                                                  2.0F / 16, 1.0F / 16, 2.0F / 16, 1.0F / 16};

  const int expanded_cols = block.expanded_cols;

  for (int ch = 0; ch < channels_; ++ch) {
    for (int i = overlap_; i < block.expanded_rows - overlap_; ++i) {
      for (int j = overlap_; j < block.expanded_cols - overlap_; ++j) {
        float sum = 0.0F;
        const float *kernel_ptr = kErnel.data();

        for (int ki = -1; ki <= 1; ++ki) {
          for (int kj = -1; kj <= 1; ++kj) {
            const int idx = (((i + ki) * expanded_cols + (j + kj)) * channels_) + ch;
            sum += static_cast<float>(my_block[idx]) * (*kernel_ptr);
            ++kernel_ptr;
          }
        }

        sum = std::max(0.0F, std::min(255.0F, sum));
        const int dst_idx = ((i * expanded_cols + j) * channels_) + ch;
        filtered_block[dst_idx] = static_cast<unsigned char>(std::lround(sum));
      }
    }
  }
}
void PapulinaYGaussFilterMPI::GetResult(const int &rank, const Block &block,
                                        const std::vector<unsigned char> &filtered_block) {
  std::vector<unsigned char> my_result(static_cast<size_t>(block.my_block_rows * block.my_block_cols * channels_));
  ExtractBlock(block, filtered_block, my_result);
  std::vector<unsigned char> final_image(static_cast<size_t>(height_ * width_ * channels_), 0);
  if (rank == 0) {
    FillImage(block, my_result, final_image);
    for (int src = 1; src < procNum_; src++) {
      int src_row = src / grid_cols_;
      int src_col = src % grid_cols_;

      int src_block_rows = block_rows_ + (src_row < extra_rows_ ? 1 : 0);
      int src_block_cols = block_cols_ + (src_col < extra_cols_ ? 1 : 0);

      int src_start_row = (src_row * block_rows_) + std::min(src_row, extra_rows_);
      int src_start_col = (src_col * block_cols_) + std::min(src_col, extra_cols_);

      Block src_block(src_block_rows, src_block_cols, src_block_rows, src_block_cols, src_start_row, src_start_col);

      std::vector<unsigned char> src_result(static_cast<size_t>(src_block_rows * src_block_cols * channels_));

      MPI_Recv(src_result.data(), src_block_rows * src_block_cols * channels_, MPI_UNSIGNED_CHAR, src, 0,
               MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      FillImage(src_block, src_result, final_image);
    }
  } else {
    MPI_Send(my_result.data(), block.my_block_rows * block.my_block_cols * channels_, MPI_UNSIGNED_CHAR, 0, 0,
             MPI_COMM_WORLD);
  }
  MPI_Bcast(final_image.data(), height_ * width_ * channels_, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);
  GetOutput() = Picture(height_, width_, channels_, final_image);
}
void PapulinaYGaussFilterMPI::ExtractBlock(const Block &block, const std::vector<unsigned char> &filtered_block,
                                           std::vector<unsigned char> &my_result) const {
  for (int i = 0; i < block.my_block_rows; i++) {
    for (int j = 0; j < block.my_block_cols; j++) {
      for (int ch = 0; ch < channels_; ch++) {
        int src_idx = (((i + overlap_) * block.expanded_cols + (j + overlap_)) * channels_) + ch;
        int dst_idx = (((i * block.my_block_cols) + j) * channels_) + ch;
        my_result[dst_idx] = filtered_block[src_idx];
      }
    }
  }
}
void PapulinaYGaussFilterMPI::FillImage(const Block &block, const std::vector<unsigned char> &my_result,
                                        std::vector<unsigned char> &final_image) const {
  for (int i = 0; i < block.my_block_rows; i++) {
    for (int j = 0; j < block.my_block_cols; j++) {
      int global_i = block.start_row + i;
      int global_j = block.start_col + j;

      for (int ch = 0; ch < channels_; ch++) {
        int src_idx = ((i * block.my_block_cols + j) * channels_) + ch;
        int dst_idx = ((global_i * width_ + global_j) * channels_) + ch;
        final_image[dst_idx] = my_result[src_idx];
      }
    }
  }
}
bool PapulinaYGaussFilterMPI::PostProcessingImpl() {
  return true;
}

}  // namespace papulina_y_gauss_filter_block
