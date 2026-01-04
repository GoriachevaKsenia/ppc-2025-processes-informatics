#include "gusev_d_star/seq/include/ops_seq.hpp"

#include "gusev_d_star/common/include/common.hpp"

namespace gusev_d_star {

GusevDStarSEQ::GusevDStarSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool GusevDStarSEQ::ValidationImpl() {
  return (GetInput() > 0);
}

bool GusevDStarSEQ::PreProcessingImpl() {
  GetOutput() = 2 * GetInput();
  return true;
}

bool GusevDStarSEQ::RunImpl() {
  auto n = GetInput();
  if (n == 0) {
    return false;
  }

  volatile int local_res = 0;

  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      for (int k = 0; k < n; k++) {
        local_res += (i + j + k);
        local_res -= (i + j + k);
      }
    }
  }

  GetOutput() += local_res;
  return true;
}

bool GusevDStarSEQ::PostProcessingImpl() {
  GetOutput() -= GetInput();
  return true;
}

}  // namespace gusev_d_star
