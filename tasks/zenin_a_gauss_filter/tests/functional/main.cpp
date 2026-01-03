#include <gtest/gtest.h>
#include <stb/stb_image.h>

#include <array>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <random>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"
#include "zenin_a_gauss_filter/common/include/common.hpp"
#include "zenin_a_gauss_filter/mpi/include/ops_mpi.hpp"
#include "zenin_a_gauss_filter/seq/include/ops_seq.hpp"

namespace zenin_a_gauss_filter {

class ZeninAGaussFilterFunctTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return std::get<0>(test_param);
  }

 protected:
  void SetUp() override {
    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());

    const std::string test_name = std::get<0>(params);
    const int width = std::get<1>(params);
    const int height = std::get<2>(params);
    const int channels = std::get<3>(params);

    std::mt19937 gen(static_cast<std::mt19937::result_type>(std::hash<std::string>{}(test_name)));
    std::uniform_int_distribution<int> dist(0, 255);

    std::vector<std::uint8_t> pixels(static_cast<std::size_t>(width) * height * channels);
    for (std::size_t i = 0; i < pixels.size(); ++i) {
      pixels[i] = static_cast<std::uint8_t>(dist(gen));
    }

    input_data_ = Image(height, width, channels, std::move(pixels));

    ZeninAGaussFilterSEQ seq_filter(input_data_);
    ASSERT_TRUE(seq_filter.Validation());
    ASSERT_TRUE(seq_filter.PreProcessing());
    ASSERT_TRUE(seq_filter.Run());
    ASSERT_TRUE(seq_filter.PostProcessing());

    expected_pixels_ = seq_filter.GetOutput().pixels;
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return output_data.pixels == expected_pixels_;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
  std::vector<std::uint8_t> expected_pixels_;
};

namespace {

TEST_P(ZeninAGaussFilterFunctTests, Test) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 44> kTestParam = {

    std::make_tuple("GradXBW", 64, 8, 1),
    std::make_tuple("GradYBW", 8, 64, 1),
    std::make_tuple("GradXRGB", 63, 19, 3),
    std::make_tuple("GradYRGB", 19, 63, 3),
    std::make_tuple("ImpulseCenterBW", 33, 33, 1),
    std::make_tuple("ImpulseCornerBW", 33, 33, 1),
    std::make_tuple("StepVertBW", 50, 20, 1),
    std::make_tuple("StepHorRGB", 45, 23, 3),
    std::make_tuple("3x3BW", 3, 3, 1),
    std::make_tuple("4x4BW", 4, 4, 1),
    std::make_tuple("5x5BW", 5, 5, 1),
    std::make_tuple("6x6BW", 6, 6, 1),

    std::make_tuple("3x3RGB", 3, 3, 3),
    std::make_tuple("4x4RGB", 4, 4, 3),
    std::make_tuple("5x5RGB", 5, 5, 3),

    std::make_tuple("10x10BW", 10, 10, 1),
    std::make_tuple("15x15RGB", 15, 15, 3),
    std::make_tuple("OnePixelBW", 1, 1, 1),
    std::make_tuple("OnePixelRGB", 1, 1, 3),
    std::make_tuple("OneRow3ColsBW", 3, 1, 1),
    std::make_tuple("OneCol3RowsBW", 1, 3, 1),
    std::make_tuple("WideBW", 50, 3, 1),
    std::make_tuple("TallBW", 3, 50, 1),

    std::make_tuple("7x7BW", 7, 7, 1),
    std::make_tuple("11x11BW", 11, 11, 1),
    std::make_tuple("13x13RGB", 13, 13, 3),
    std::make_tuple("17x17BW", 17, 17, 1),
    std::make_tuple("19x19BW", 19, 19, 1),
    std::make_tuple("23x23BW", 23, 23, 1),
    std::make_tuple("29x29BW", 29, 29, 1),
    std::make_tuple("31x31BW", 31, 31, 1),

    std::make_tuple("Rect3x7BW", 3, 7, 1),
    std::make_tuple("Rect7x3BW", 7, 3, 1),
    std::make_tuple("Rect5x8RGB", 5, 8, 3),
    std::make_tuple("Rect8x5RGB", 8, 5, 3),

    std::make_tuple("Prime53x53BW", 53, 53, 1),
    std::make_tuple("Prime59x59BW", 59, 59, 1),
    std::make_tuple("Prime61x61RGB", 61, 61, 3),
    std::make_tuple("Prime67x67BW", 67, 67, 1),
    std::make_tuple("Const0BW", 64, 64, 1),
    std::make_tuple("Const255RGB", 32, 32, 3),
    std::make_tuple("Const17BW", 17, 19, 1),

    std::make_tuple("CheckerBW", 31, 31, 1),
    std::make_tuple("CheckerRGB", 32, 17, 3),

};

const auto kTestTasksList =
    std::tuple_cat(ppc::util::AddFuncTask<ZeninAGaussFilterMPI, InType>(kTestParam, PPC_SETTINGS_zenin_a_gauss_filter),
                   ppc::util::AddFuncTask<ZeninAGaussFilterSEQ, InType>(kTestParam, PPC_SETTINGS_zenin_a_gauss_filter));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);
const auto kPerfTestName = ZeninAGaussFilterFunctTests::PrintFuncTestName<ZeninAGaussFilterFunctTests>;

INSTANTIATE_TEST_SUITE_P(ZeninAGaussFilter, ZeninAGaussFilterFunctTests, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace zenin_a_gauss_filter
