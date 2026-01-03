#include "nikolaev_d_gather/seq/include/ops_seq.hpp"

#include <mpi.h>

#include <cstddef>
#include <vector>

#include "nikolaev_d_gather/common/include/common.hpp"

namespace nikolaev_d_gather {

namespace {
size_t GetTypeSizeSeq(MPI_Datatype datatype) {
  if (datatype == MPI_INT) {
    return sizeof(int);
  }
  if (datatype == MPI_FLOAT) {
    return sizeof(float);
  }
  if (datatype == MPI_DOUBLE) {
    return sizeof(double);
  }
  return 0;
}
}  // namespace

NikolaevDGatherSEQ::NikolaevDGatherSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

bool NikolaevDGatherSEQ::ValidationImpl() {
  const auto &input = GetInput();

  if (input.data.empty()) {
    return false;
  }
  if (input.count <= 0) {
    return false;
  }

  if (input.root < 0) {
    return false;
  }

  size_t type_size = GetTypeSizeSeq(input.datatype);

  if (type_size == 0) {
    return false;
  }

  if (input.data.size() != static_cast<size_t>(input.count) * type_size) {
    return false;
  }

  return true;
}

bool NikolaevDGatherSEQ::PreProcessingImpl() {
  return true;
}

bool NikolaevDGatherSEQ::RunImpl() {
  const auto &input = GetInput();

  GetOutput() = input.data;

  return true;
}

bool NikolaevDGatherSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace nikolaev_d_gather
