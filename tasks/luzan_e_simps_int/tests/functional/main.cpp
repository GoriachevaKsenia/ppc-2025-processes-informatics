#include <gtest/gtest.h>
#include <stb/stb_image.h>

#include <array>
#include <cmath>
#include <string>
#include <tuple>

#include "luzan_e_simps_int/common/include/common.hpp"
#include "luzan_e_simps_int/mpi/include/ops_mpi.hpp"
#include "luzan_e_simps_int/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace luzan_e_simps_int {

class LuzanESimpsIntFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    int n = get<0>(test_param);
    auto a = static_cast<unsigned>(100.0 * fabs(std::get<0>(std::get<1>(test_param))));
    auto b = static_cast<unsigned>(100.0 * fabs(std::get<1>(std::get<1>(test_param))));
    auto c = static_cast<unsigned>(100.0 * fabs(std::get<0>(std::get<2>(test_param))));
    auto d = static_cast<unsigned>(100.0 * fabs(std::get<1>(std::get<2>(test_param))));
    int func_num = std::get<3>(test_param);

    std::string rectangle =
        "rectangle_" + std::to_string(a) + "_" + std::to_string(b) + "__" + std::to_string(c) + "_" + std::to_string(d);
    return "for_" + std::to_string(n) + "pieces_" + rectangle + "__by_func_No_" + std::to_string(func_num);
  }

 protected:
  void SetUp() override {
    TestType params = std::get<static_cast<int>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());

    int n = std::get<0>(params);
    auto ab = std::get<1>(params);
    auto cd = std::get<2>(params);
    int fn = std::get<3>(params);

    input_data_ = std::make_tuple(n, ab, cd, fn);
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

 private:
  InType input_data_;
};

namespace {

TEST_P(LuzanESimpsIntFuncTests, MatmulFromPic) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 16> kTestParam = {
    std::make_tuple(4, std::make_tuple(3, 5), std::make_tuple(3, 5), 1),          // std
    std::make_tuple(4, std::make_tuple(3, 4), std::make_tuple(2, 3), 5),          // std
    std::make_tuple(10, std::make_tuple(0.5, 1), std::make_tuple(0.5, 1), 2),     // small rectangle
    std::make_tuple(70, std::make_tuple(-50, 50), std::make_tuple(-50, 50), 1),   // big rectangle
    std::make_tuple(100, std::make_tuple(-100, 0), std::make_tuple(0, 1), 1),     // tube
    std::make_tuple(100, std::make_tuple(-10, 0), std::make_tuple(0, 100), 1),    // vertical tube
    std::make_tuple(6, std::make_tuple(3, 5), std::make_tuple(3, 5), 2),          // std
    std::make_tuple(24, std::make_tuple(-2, 5), std::make_tuple(12, 25), 3),      // std
    std::make_tuple(70, std::make_tuple(-10, -5), std::make_tuple(-50, -15), 4),  // negative rectangle
    std::make_tuple(70, std::make_tuple(-10, -5), std::make_tuple(15, 25), 4),    // half negative rectangle
    std::make_tuple(70, std::make_tuple(5, 10), std::make_tuple(-25, -5), 4),     // half negative rectangle 2
    std::make_tuple(10, std::make_tuple(-60, 60), std::make_tuple(-60, 60), 1),   // big rectangle, small n
    std::make_tuple(200, std::make_tuple(-2, 5), std::make_tuple(12, 25), 3),     // small rectangle, big n
    std::make_tuple(30, std::make_tuple(-20, 5), std::make_tuple(2, 25), 4),      // std
    std::make_tuple(300, std::make_tuple(0, 1), std::make_tuple(0, 1), 4),        // small rectangle, big n
    std::make_tuple(6, std::make_tuple(0, 1), std::make_tuple(0, 1), 1)           // small rectangle, small n
};

const auto kTestTasksList =
    std::tuple_cat(ppc::util::AddFuncTask<LuzanESimpsIntMPI, InType>(kTestParam, PPC_SETTINGS_luzan_e_simps_int),
                   ppc::util::AddFuncTask<LuzanESimpsIntSEQ, InType>(kTestParam, PPC_SETTINGS_luzan_e_simps_int));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName = LuzanESimpsIntFuncTests::PrintFuncTestName<LuzanESimpsIntFuncTests>;

INSTANTIATE_TEST_SUITE_P(PicMatrixTests, LuzanESimpsIntFuncTests, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace luzan_e_simps_int
