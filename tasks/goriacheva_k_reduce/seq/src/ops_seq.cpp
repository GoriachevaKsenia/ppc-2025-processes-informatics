#include "goriacheva_k_reduce//seq/include/ops_seq.hpp"

#include <numeric>

#include "goriacheva_k_reduce//common/include/common.hpp"

namespace goriacheva_k_reduce {

GoriachevaKReduceSEQ::GoriachevaKReduceSEQ(const InType &in) {
  InType tmp(in);
  GetInput().swap(tmp);
  SetTypeOfTask(GetStaticTypeOfTask());
}

bool GoriachevaKReduceSEQ::ValidationImpl() {
  return true;
}

bool GoriachevaKReduceSEQ::PreProcessingImpl() {
  GetOutput().resize(1);
  return true;
}

bool GoriachevaKReduceSEQ::RunImpl() {
  const auto input = GetInput();
  if (input.empty()) {
    GetOutput()[0] = 0;
    return true;
  }
  GetOutput()[0] = std::accumulate(GetInput().begin(), GetInput().end(), 0);
  return true;
}

bool GoriachevaKReduceSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace goriacheva_k_reduce
