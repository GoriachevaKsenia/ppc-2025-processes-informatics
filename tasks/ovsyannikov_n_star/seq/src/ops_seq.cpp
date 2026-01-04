#include "ovsyannikov_n_star/seq/include/ops_seq.hpp"

#include "ovsyannikov_n_star/common/include/common.hpp"

namespace ovsyannikov_n_star {

OvsyannikovNStarSEQ::OvsyannikovNStarSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool OvsyannikovNStarSEQ::ValidationImpl() {
  return true;
}

bool OvsyannikovNStarSEQ::PreProcessingImpl() {
  GetOutput() = 0;
  return true;
}

bool OvsyannikovNStarSEQ::RunImpl() {
  GetOutput() = GetInput();
  return true;
}

bool OvsyannikovNStarSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace ovsyannikov_n_star
