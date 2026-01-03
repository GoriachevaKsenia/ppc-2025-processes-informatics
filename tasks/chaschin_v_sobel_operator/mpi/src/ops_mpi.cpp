#include "chaschin_v_sobel_operator/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <tuple>
#include <utility>
#include <vector>

#include "chaschin_v_sobel_operator/common/include/common.hpp"

namespace chaschin_v_sobel_operator {

ChaschinVSobelOperatorMPI::ChaschinVSobelOperatorMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  auto in_copy = in;

  GetInput() = std::move(in_copy);
  GetOutput().clear();

  int height = std::get<1>(in);
  int width = std::get<2>(in);

  Size_ = std::make_tuple(height, width);
}

bool ChaschinVSobelOperatorMPI::ValidationImpl() {
  const auto &in = GetInput();

  const auto &image = std::get<0>(in);

  return !image.empty();
}

std::vector<float> ChaschinVSobelOperatorMPI::PreprocessToGrayscaleWithOverlap(
    const std::vector<std::vector<Pixel>> &image, int n_procs, std::vector<int> &sendcounts, std::vector<int> &displs) {
  int n = static_cast<int>(image.size());
  int m = static_cast<int>(image[0].size());

  int padded_m = m + 2;

  sendcounts.resize(n_procs);
  displs.resize(n_procs, 0);
  int base = n / n_procs;
  int rem = n % n_procs;

  int total_real_rows = 0;
  std::vector<int> l_r(n_procs + 2, 0);
  for (int rank = 0; rank < n_procs; ++rank) {
    bool local_tale = (rank < rem);
    int local_rows = (base + 2) + static_cast<int>(local_tale);
    sendcounts[rank] = (local_rows)*padded_m;
    displs[rank] = total_real_rows * padded_m;
    total_real_rows += local_rows;
  }

  std::vector<float> buffer((n + 2 + (static_cast<size_t>(n_procs) - 1) * 2) * static_cast<size_t>(padded_m), 0.0F);

  for (int rank = 0; rank < n_procs + 1; ++rank) {
    bool local_tale = (rank < rem);
    l_r[rank + 1] = l_r[rank] + (base) + static_cast<int>(local_tale);
  }

  for (int ii = 0; ii < n_procs; ii++) {
    for (int xx = -1; xx < (sendcounts[ii] / padded_m) - 1; xx++) {
      for (int yy = 0; yy < m; yy++) {
        float val = 0.0F;
        if ((xx + l_r[ii] >= 0) && (xx + l_r[ii] < n)) {
          const Pixel &p = image[xx + l_r[ii]][yy];
          val = (0.299F * static_cast<float>(p.r)) + (0.587F * static_cast<float>(p.g)) +
                (0.114F * static_cast<float>((p.b)));
        }
        buffer[displs[ii] + ((xx + 1) * padded_m) + yy + 1] = val;
      }
    }
  }
  return buffer;
}

bool ChaschinVSobelOperatorMPI::PreProcessingImpl() {
  const auto &in = GetInput();

  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if (rank == 0) {
    const auto &image = std::get<0>(in);

    int n_procs = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &n_procs);

    std::vector<int> sendcounts;
    std::vector<int> displs;

    PreProcessGray_ = PreprocessToGrayscaleWithOverlap(image, n_procs, sendcounts, displs);

    ScatterSendCounts_ = sendcounts;
    ScatterDispls_ = displs;
  }

  return true;
}

float ChaschinVSobelOperatorMPI::SobelAt(const std::vector<float> &img, int i, int j, int stride) {
  std::vector<float> k_kx = {-1.0F, 0.0F, 1.0F, -2.0F, 0.0F, 2.0F, -1.0F, 0.0F, 1.0F};
  std::vector<float> k_ky = {-1.0F, -2.0F, -1.0F, 0.0F, 0.0F, 0.0F, 1.0F, 2.0F, 1.0F};

  float gx = 0.0F;
  float gy = 0.0F;

  for (int di = -1; di <= 1; ++di) {
    int ni = i + di;
    for (int dj = -1; dj <= 1; ++dj) {
      int nj = j + dj;
      auto idx = (static_cast<size_t>(di + 1) * static_cast<size_t>(3)) + static_cast<size_t>(dj + 1);

      float val = img[(static_cast<size_t>(ni) * stride) + static_cast<size_t>(nj)];
      gx += val * k_kx[idx];
      gy += val * k_ky[idx];
    }
  }

  return std::sqrt((gx * gx) + (gy * gy));
}

bool ChaschinVSobelOperatorMPI::RunImpl() {
  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  int n = std::get<0>(Size_);
  int m = std::get<1>(Size_);

  const auto &in = PreProcessGray_;

  int base = n / size;
  int rem = n % size;
  int padded_m = m + 2;

  if (rank != 0) {
    ScatterSendCounts_.resize(size);
  }

  MPI_Bcast(ScatterSendCounts_.data(), size, MPI_INT, 0, MPI_COMM_WORLD);
  int recvcount = ScatterSendCounts_[rank];
  int local_rows = recvcount / padded_m;
  std::vector<float> local_block;
  local_block.resize(recvcount);

  MPI_Scatterv(rank == 0 ? in.data() : nullptr, ScatterSendCounts_.data(), ScatterDispls_.data(), MPI_FLOAT,
               local_block.data(), recvcount, MPI_FLOAT, 0, MPI_COMM_WORLD);

  std::vector<float> local_output(static_cast<size_t>(local_rows - 2) * static_cast<size_t>(m));

  for (int i = 1; i <= local_rows - 2; ++i) {
    for (int j = 1; j <= m; ++j) {
      local_output[((i - 1) * m) + (j - 1)] = SobelAt(local_block, i, j, padded_m);
    }
  }

  std::vector<int> recvcounts(size);
  std::vector<int> displs_out(size);
  int offset_res = 0;
  for (int pp = 0; pp < size; ++pp) {
    int rows_p = base + (pp < rem ? 1 : 0);
    recvcounts[pp] = rows_p * m;
    displs_out[pp] = offset_res;
    offset_res += recvcounts[pp];
  }

  if (rank == 0) {
    PostProcessGray_.resize(static_cast<size_t>(n) * static_cast<size_t>(m));
  }

  MPI_Gatherv(local_output.data(), (local_rows - 2) * m, MPI_FLOAT, rank == 0 ? PostProcessGray_.data() : nullptr,
              recvcounts.data(), displs_out.data(), MPI_FLOAT, 0, MPI_COMM_WORLD);

  return true;
}

bool ChaschinVSobelOperatorMPI::PostProcessingImpl() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  auto &out = GetOutput();
  int n = std::get<0>(Size_);
  int m = std::get<1>(Size_);
  if (rank != 0) {
    PostProcessGray_.resize(static_cast<size_t>(n) * static_cast<size_t>(m));
  }

  MPI_Bcast(PostProcessGray_.data(), n * m, MPI_FLOAT, 0, MPI_COMM_WORLD);

  out.resize(n);

  for (int i = 0; i < n; ++i) {
    out[i].resize(m);
    for (int j = 0; j < m; ++j) {
      float v = PostProcessGray_[(i * m) + j];
      unsigned char c = static_cast<unsigned char>(std::clamp(v, 0.0F, 255.0F));
      out[i][j] = Pixel{.r = c, .g = c, .b = c};
    }
  }

  return true;
}

}  // namespace chaschin_v_sobel_operator
