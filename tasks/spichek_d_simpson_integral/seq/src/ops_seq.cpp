#include "spichek_d_simpson_integral/seq/include/ops_seq.hpp"

#include <cmath>

// Explicitly include common.hpp to satisfy clang-tidy misc-include-cleaner for InType/OutType
#include "spichek_d_simpson_integral/common/include/common.hpp"

namespace spichek_d_simpson_integral {

namespace {
// Helper to reduce cognitive complexity and avoid nested ternary operators
int GetSimpsonWeight(int index, int n) {
  if (index == 0 || index == n) {
    return 1;
  }
  return (index % 2 == 0) ? 2 : 4;
}
}  // namespace

SpichekDSimpsonIntegralSEQ::SpichekDSimpsonIntegralSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool SpichekDSimpsonIntegralSEQ::ValidationImpl() {
  return (GetInput() > 0) && (GetInput() % 2 == 0);
}

bool SpichekDSimpsonIntegralSEQ::PreProcessingImpl() {
  GetOutput() = 0;
  return true;
}

bool SpichekDSimpsonIntegralSEQ::RunImpl() {
  int n = GetInput();
  double h = 1.0 / n;
  double sum = 0.0;

  for (int i = 0; i <= n; i++) {
    const double x = i * h;
    const int wx = GetSimpsonWeight(i, n);

    for (int j = 0; j <= n; j++) {
      const double y = j * h;
      const int wy = GetSimpsonWeight(j, n);

      sum += wx * wy * (x * x + y * y);
    }
  }

  double result = sum * h * h / 9.0;
  GetOutput() = static_cast<OutType>(std::round(result));
  return true;
}

bool SpichekDSimpsonIntegralSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace spichek_d_simpson_integral
