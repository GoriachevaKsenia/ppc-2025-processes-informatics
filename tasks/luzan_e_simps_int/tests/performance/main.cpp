#include <gtest/gtest.h>

#include <cmath>
#include <tuple>

#include "luzan_e_simps_int/common/include/common.hpp"
#include "luzan_e_simps_int/mpi/include/ops_mpi.hpp"
#include "luzan_e_simps_int/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace luzan_e_simps_int {

class LuzanESimpsIntpERFTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
  const double a_ = -100.0;
  const double b_ = 100.0;
  const double c_ = -200.0;
  const double d_ = 200.0;
  const int n_ = 400;
  const int func_num_ = 5;
  InType input_data_;

  void SetUp() override {
    input_data_ = std::make_tuple(n_, std::make_tuple(a_, b_), std::make_tuple(c_, d_), func_num_);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    double a = 0.0;
    double b = 0.0;
    double c = 0.0;
    double d = 0.0;
    int n = 0;  // кол-во отрезков
    int func_num = 0;

    // getting data
    n = std::get<0>(input_data_);
    a = std::get<0>(std::get<1>(input_data_));
    b = std::get<1>(std::get<1>(input_data_));
    c = std::get<0>(std::get<2>(input_data_));
    d = std::get<1>(std::get<2>(input_data_));
    func_num = std::get<3>(input_data_);

    double (*fp)(double, double) = GetFunc(func_num);
    double hx = (b - a) / n;
    double hy = (d - c) / n;

    double sum = 0;
    double wx = 1.0;
    double wy = 1.0;
    double x = 0.0;
    double y = 0.0;

    for (int i = 0; i <= n; i++) {
      x = a + (hx * i);
      wx = GetWeight(i, n);

      for (int j = 0; j <= n; j++) {
        y = c + (hy * j);
        wy = GetWeight(j, n);
        sum += wy * wx * fp(x, y);
      }
    }
    sum = sum * hx * hy / (3 * 3);

    return (fabs(output_data - sum) < kEpsilon);
  }

  InType GetTestInputData() final {
    return input_data_;
  }
};

TEST_P(LuzanESimpsIntpERFTests, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, LuzanESimpsIntMPI, LuzanESimpsIntSEQ>(PPC_SETTINGS_luzan_e_simps_int);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = LuzanESimpsIntpERFTests::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, LuzanESimpsIntpERFTests, kGtestValues, kPerfTestName);

}  // namespace luzan_e_simps_int
