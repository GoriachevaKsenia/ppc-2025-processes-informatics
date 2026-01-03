#include "zenin_a_gauss_filter/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <utility>
#include <vector>

#include "zenin_a_gauss_filter/common/include/common.hpp"

namespace zenin_a_gauss_filter {

namespace {

constexpr int kHalo = 1;
constexpr int kTagExpanded = 200;
constexpr int kTagResult = 500;

struct BlockInfo {
  int my_h = 0, my_w = 0;
  int start_y = 0, start_x = 0;
};

std::size_t GlobalIdx(int gx, int gy, int chan, int width, int channels) {
  return ((static_cast<std::size_t>(gy) * width + gx) * channels) + static_cast<std::size_t>(chan);
}

int Clampi(int v, int lo, int hi) {
  return std::max(lo, std::min(hi, v));
}

std::uint8_t Clampu8(int v) {
  return static_cast<std::uint8_t>(Clampi(v, 0, 255));
}

std::uint8_t GetLocal(const std::vector<std::uint8_t> &buf, int local_w_with_halo, int ch, int x, int y, int c) {
  const int idx = ((y * local_w_with_halo + x) * ch) + c;
  return buf[idx];
}

BlockInfo CalcBlock(int pr, int pc, int h, int w, int grid_r, int grid_c) {
  const int base_h = h / grid_r;
  const int base_w = w / grid_c;
  const int extra_h = h % grid_r;
  const int extra_w = w % grid_c;

  BlockInfo b;
  b.my_h = base_h + (pr < extra_h ? 1 : 0);
  b.my_w = base_w + (pc < extra_w ? 1 : 0);

  b.start_y = (pr * base_h) + std::min(pr, extra_h);
  b.start_x = (pc * base_w) + std::min(pc, extra_w);
  return b;
}

void FillExpandedBlock(const zenin_a_gauss_filter::Image &img, const zenin_a_gauss_filter::BlockInfo &bb, int width,
                       int height, int channels, std::vector<std::uint8_t> *dst) {
  const int hh = bb.my_h;
  const int ww = bb.my_w;
  const int dst_w = ww + (2 * kHalo);
  const int dst_h = hh + (2 * kHalo);

  dst->assign(static_cast<std::size_t>(dst_h) * dst_w * channels, 0);

  for (int ly = -kHalo; ly < hh + kHalo; ++ly) {
    for (int lx = -kHalo; lx < ww + kHalo; ++lx) {
      int gy = bb.start_y + ly;
      int gx = bb.start_x + lx;

      gy = std::max(0, std::min(height - 1, gy));
      gx = std::max(0, std::min(width - 1, gx));

      const int dy = ly + kHalo;
      const int dx = lx + kHalo;

      for (int chan = 0; chan < channels; ++chan) {
        (*dst)[((dy * dst_w + dx) * channels) + chan] = img.pixels[GlobalIdx(gx, gy, chan, width, channels)];
      }
    }
  }
}

void BuildOrRecvExpandedBlock(int rank, int proc_num, int grid_cols, int width, int height, int channels,
                              const zenin_a_gauss_filter::BlockInfo &my_block,
                              const std::function<zenin_a_gauss_filter::BlockInfo(int, int)> &calc_block,
                              const zenin_a_gauss_filter::Image *root_img, std::vector<std::uint8_t> *local_in) {
  if (rank == 0) {
    FillExpandedBlock(*root_img, my_block, width, height, channels, local_in);

    for (int rnk = 1; rnk < proc_num; ++rnk) {
      const int rpr = rnk / grid_cols;
      const int rpc = rnk % grid_cols;
      const auto rb = calc_block(rpr, rpc);

      std::vector<std::uint8_t> pack;
      FillExpandedBlock(*root_img, rb, width, height, channels, &pack);
      MPI_Send(pack.data(), static_cast<int>(pack.size()), MPI_UNSIGNED_CHAR, rnk, kTagExpanded, MPI_COMM_WORLD);
    }
  } else {
    MPI_Recv(local_in->data(), static_cast<int>(local_in->size()), MPI_UNSIGNED_CHAR, 0, kTagExpanded, MPI_COMM_WORLD,
             MPI_STATUS_IGNORE);
  }
}

void ConvolveLocalBlock(const std::vector<std::uint8_t> &local_in, int lw, int my_w, int my_h, int channels,
                        std::vector<std::uint8_t> *local_out) {
  constexpr int kKernelSum = 16;

  for (int yd = 0; yd < my_h; ++yd) {
    const int ly = yd + kHalo;
    for (int xd = 0; xd < my_w; ++xd) {
      const int lx = xd + kHalo;
      for (int chan = 0; chan < channels; ++chan) {
        const int v00 = static_cast<int>(zenin_a_gauss_filter::GetLocal(local_in, lw, channels, lx - 1, ly - 1, chan));
        const int v01 = static_cast<int>(zenin_a_gauss_filter::GetLocal(local_in, lw, channels, lx, ly - 1, chan));
        const int v02 = static_cast<int>(zenin_a_gauss_filter::GetLocal(local_in, lw, channels, lx + 1, ly - 1, chan));

        const int v10 = static_cast<int>(zenin_a_gauss_filter::GetLocal(local_in, lw, channels, lx - 1, ly, chan));
        const int v11 = static_cast<int>(zenin_a_gauss_filter::GetLocal(local_in, lw, channels, lx, ly, chan));
        const int v12 = static_cast<int>(zenin_a_gauss_filter::GetLocal(local_in, lw, channels, lx + 1, ly, chan));

        const int v20 = static_cast<int>(zenin_a_gauss_filter::GetLocal(local_in, lw, channels, lx - 1, ly + 1, chan));
        const int v21 = static_cast<int>(zenin_a_gauss_filter::GetLocal(local_in, lw, channels, lx, ly + 1, chan));
        const int v22 = static_cast<int>(zenin_a_gauss_filter::GetLocal(local_in, lw, channels, lx + 1, ly + 1, chan));

        int sum = 0;
        sum += v00 * 1;
        sum += v01 * 2;
        sum += v02 * 1;
        sum += v10 * 2;
        sum += v11 * 4;
        sum += v12 * 2;
        sum += v20 * 1;
        sum += v21 * 2;
        sum += v22 * 1;

        const int res = (sum + (kKernelSum / 2)) / kKernelSum;
        (*local_out)[((yd * my_w + xd) * channels) + chan] = zenin_a_gauss_filter::Clampu8(res);
      }
    }
  }
}

void CopyBlockToImage(const BlockInfo &block, const std::vector<std::uint8_t> &src, int src_w, int width, int channels,
                      std::vector<std::uint8_t> *dst) {
  for (int yd = 0; yd < block.my_h; ++yd) {
    for (int xd = 0; xd < block.my_w; ++xd) {
      const int gy = block.start_y + yd;
      const int gx = block.start_x + xd;
      for (int chan = 0; chan < channels; ++chan) {
        (*dst)[((gy * width + gx) * channels) + chan] = src[((yd * src_w + xd) * channels) + chan];
      }
    }
  }
}

void GatherAndBroadcastResult(int rank, int proc_num, int grid_cols, int width, int channels, const BlockInfo &my_block,
                              const std::function<BlockInfo(int, int)> &calc_block,
                              const std::vector<std::uint8_t> &local_out, std::vector<std::uint8_t> *final_image) {
  if (rank == 0) {
    CopyBlockToImage(my_block, local_out, my_block.my_w, width, channels, final_image);

    for (int src_rank = 1; src_rank < proc_num; ++src_rank) {
      const int spr = src_rank / grid_cols;
      const int spc = src_rank % grid_cols;
      const BlockInfo sb = calc_block(spr, spc);

      std::vector<std::uint8_t> recv(static_cast<std::size_t>(sb.my_h) * sb.my_w * channels);
      MPI_Recv(recv.data(), static_cast<int>(recv.size()), MPI_UNSIGNED_CHAR, src_rank, kTagResult, MPI_COMM_WORLD,
               MPI_STATUS_IGNORE);

      CopyBlockToImage(sb, recv, sb.my_w, width, channels, final_image);
    }
  } else {
    MPI_Send(local_out.data(), static_cast<int>(local_out.size()), MPI_UNSIGNED_CHAR, 0, kTagResult, MPI_COMM_WORLD);
  }

  MPI_Bcast(final_image->data(), static_cast<int>(final_image->size()), MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);
}

}  // namespace

ZeninAGaussFilterMPI::ZeninAGaussFilterMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = OutType{};
}

bool ZeninAGaussFilterMPI::ValidationImpl() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  bool ok = true;
  if (rank == 0) {
    const auto &in = GetInput();
    const std::size_t need = static_cast<std::size_t>(in.width) * in.height * in.channels;

    ok = (in.width > 0) && (in.height > 0) && (in.channels == 1 || in.channels == 3) && (in.pixels.size() == need);
  }

  int ok_int = ok ? 1 : 0;

  MPI_Bcast(&ok_int, 1, MPI_INT, 0, MPI_COMM_WORLD);
  return ok_int == 1;
}

bool ZeninAGaussFilterMPI::PreProcessingImpl() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &proc_num_);

  if (rank == 0) {
    const auto &in = GetInput();
    width_ = in.width;
    height_ = in.height;
    channels_ = in.channels;
  }

  MPI_Bcast(&width_, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&height_, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&channels_, 1, MPI_INT, 0, MPI_COMM_WORLD);

  std::array<int, 2> dims{0, 0};
  MPI_Dims_create(proc_num_, 2, dims.data());
  grid_rows_ = dims[0];
  grid_cols_ = dims[1];

  block_h_ = height_ / grid_rows_;
  block_w_ = width_ / grid_cols_;
  extra_h_ = height_ % grid_rows_;
  extra_w_ = width_ % grid_cols_;

  if (rank == 0) {
    auto &out = GetOutput();
    out.height = height_;
    out.width = width_;
    out.channels = channels_;
    out.pixels.assign(static_cast<std::size_t>(width_) * height_ * channels_, 0);
  }

  return true;
}

bool ZeninAGaussFilterMPI::RunImpl() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  const int pr = rank / grid_cols_;
  const int pc = rank % grid_cols_;

  const BlockInfo my_block = CalcBlock(pr, pc, height_, width_, grid_rows_, grid_cols_);
  const int my_h = my_block.my_h;
  const int my_w = my_block.my_w;

  const int lw = my_w + (2 * kHalo);
  const int lh = my_h + (2 * kHalo);

  std::vector<std::uint8_t> local_in(static_cast<std::size_t>(lh) * lw * channels_, 0);
  std::vector<std::uint8_t> local_out(static_cast<std::size_t>(my_h) * my_w * channels_, 0);

  auto calc_block = [&](int rpr, int rpc) -> BlockInfo {
    return CalcBlock(rpr, rpc, height_, width_, grid_rows_, grid_cols_);
  };

  if (rank == 0) {
    const auto &img = GetInput();
    BuildOrRecvExpandedBlock(rank, proc_num_, grid_cols_, width_, height_, channels_, my_block, calc_block, &img,
                             &local_in);
  } else {
    BuildOrRecvExpandedBlock(rank, proc_num_, grid_cols_, width_, height_, channels_, my_block, calc_block, nullptr,
                             &local_in);
  }

  ConvolveLocalBlock(local_in, lw, my_w, my_h, channels_, &local_out);

  std::vector<std::uint8_t> final_image(static_cast<std::size_t>(width_) * height_ * channels_, 0);
  GatherAndBroadcastResult(rank, proc_num_, grid_cols_, width_, channels_, my_block, calc_block, local_out,
                           &final_image);

  GetOutput() = OutType{height_, width_, channels_, std::move(final_image)};
  return true;
}

bool ZeninAGaussFilterMPI::PostProcessingImpl() {
  return true;
}

}  // namespace zenin_a_gauss_filter
