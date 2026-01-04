#include "lifanov_k_trapezoid_method/seq/include/ops_seq.hpp"

#include "lifanov_k_trapezoid_method/common/include/common.hpp"

namespace {
double Function(double x, double y) {
  return (x * x) + (y * y);
}
}  // namespace

namespace lifanov_k_trapezoid_method {

LifanovKTrapezoidMethodSEQ::LifanovKTrapezoidMethodSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0.0;
}

bool LifanovKTrapezoidMethodSEQ::ValidationImpl() {
  const auto &in = GetInput();

  if (in.size() != 6) {
    return false;
  }

  const double ax = in[0];
  const double bx = in[1];
  const double ay = in[2];
  const double by = in[3];
  const int nx = static_cast<int>(in[4]);
  const int ny = static_cast<int>(in[5]);

  return (bx > ax) && (by > ay) && (nx > 0) && (ny > 0);
}

bool LifanovKTrapezoidMethodSEQ::PreProcessingImpl() {
  GetOutput() = 0.0;
  return true;
}

bool LifanovKTrapezoidMethodSEQ::RunImpl() {
  const auto &in = GetInput();

  const double ax = in[0];
  const double bx = in[1];
  const double ay = in[2];
  const double by = in[3];
  const int nx = static_cast<int>(in[4]);
  const int ny = static_cast<int>(in[5]);

  const double hx = (bx - ax) / nx;
  const double hy = (by - ay) / ny;

  double integral = 0.0;

  for (int i = 0; i <= nx; ++i) {
    const double x = ax + (i * hx);
    const double wx = (i == 0 || i == nx) ? 0.5 : 1.0;

    for (int j = 0; j <= ny; ++j) {
      const double y = ay + (j * hy);
      const double wy = (j == 0 || j == ny) ? 0.5 : 1.0;

      integral += wx * wy * Function(x, y);
    }
  }

  GetOutput() = integral * hx * hy;
  return true;
}

bool LifanovKTrapezoidMethodSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace lifanov_k_trapezoid_method
