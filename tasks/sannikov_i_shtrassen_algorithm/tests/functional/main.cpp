#include <gtest/gtest.h>
#include <stb/stb_image.h>

#include <array>
#include <cstddef>
#include <string>
#include <tuple>
#include <vector>

#include "sannikov_i_shtrassen_algorithm/common/include/common.hpp"
#include "sannikov_i_shtrassen_algorithm/mpi/include/ops_mpi.hpp"
#include "sannikov_i_shtrassen_algorithm/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace sannikov_i_shtrassen_algorithm {

class SannikovIShtrassenAlgorithmFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return std::get<1>(test_param);
  }

 protected:
  void SetUp() override {
    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    input_data_ = std::get<0>(params);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    const auto &a = std::get<0>(input_data_);
    const auto &b = std::get<1>(input_data_);

    if (a.empty() || b.empty()) {
      return false;
    }
    if (a.size() != b.size()) {
      return false;
    }
    std::size_t n = 0;
    n = a.size();
    if (output_data.size() != n) {
      return false;
    }

    OutType ref(n, std::vector<double>(n, 0.0));
    for (std::size_t i = 0; i < n; ++i) {
      if (a[i].size() != n || b[i].size() != n || output_data[i].size() != n) {
        return false;
      }
      for (std::size_t k = 0; k < n; ++k) {
        for (std::size_t j = 0; j < n; ++j) {
          ref[i][j] += a[i][k] * b[k][j];
        }
      }
    }

    double eps = 1e-7;
    double diff = 0;
    for (std::size_t i = 0; i < n; ++i) {
      for (std::size_t j = 0; j < n; ++j) {
        diff = ref[i][j] - output_data[i][j];
        if (diff > eps || diff < -eps) {
          return false;
        }
      }
    }

    return true;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
};

namespace {

TEST_P(SannikovIShtrassenAlgorithmFuncTests, MultiplyFromParams) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 12> kTestParam = {
    std::make_tuple(InType{{{2.0}}, {{3.0}}}, "1x1_simple"),
    std::make_tuple(InType{{{-4.5}}, {{2.0}}}, "1x1_negative"),

    std::make_tuple(InType{{{1.0, 2.0}, {3.0, 4.0}}, {{5.0, 6.0}, {7.0, 8.0}}}, "2x2_basic"),

    std::make_tuple(InType{{{1.0, 0.0}, {0.0, 1.0}}, {{9.0, 8.0}, {7.0, 6.0}}}, "2x2_identity_left"),

    std::make_tuple(InType{{{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}, {7.0, 8.0, 9.0}},
                           {{9.0, 8.0, 7.0}, {6.0, 5.0, 4.0}, {3.0, 2.0, 1.0}}},
                    "3x3_basic"),

    std::make_tuple(InType{{{1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}},
                           {{2.0, 3.0, 4.0}, {5.0, 6.0, 7.0}, {8.0, 9.0, 10.0}}},
                    "3x3_identity_left"),

    std::make_tuple(
        InType{{{1.0, 2.0, 3.0, 4.0}, {5.0, 6.0, 7.0, 8.0}, {9.0, 10.0, 11.0, 12.0}, {13.0, 14.0, 15.0, 16.0}},
               {{16.0, 15.0, 14.0, 13.0}, {12.0, 11.0, 10.0, 9.0}, {8.0, 7.0, 6.0, 5.0}, {4.0, 3.0, 2.0, 1.0}}},
        "4x4_basic"),

    std::make_tuple(InType{{{1.0, 2.0, 3.0, 4.0, 5.0},
                            {6.0, 7.0, 8.0, 9.0, 10.0},
                            {11.0, 12.0, 13.0, 14.0, 15.0},
                            {16.0, 17.0, 18.0, 19.0, 20.0},
                            {21.0, 22.0, 23.0, 24.0, 25.0}},
                           {{1.0, 0.0, 0.0, 0.0, 0.0},
                            {0.0, 1.0, 0.0, 0.0, 0.0},
                            {0.0, 0.0, 1.0, 0.0, 0.0},
                            {0.0, 0.0, 0.0, 1.0, 0.0},
                            {0.0, 0.0, 0.0, 0.0, 1.0}}},
                    "5x5_identity_right"),

    std::make_tuple(InType{{{1.0, 1.0, 1.0, 1.0, 1.0, 1.0},
                            {2.0, 2.0, 2.0, 2.0, 2.0, 2.0},
                            {3.0, 3.0, 3.0, 3.0, 3.0, 3.0},
                            {4.0, 4.0, 4.0, 4.0, 4.0, 4.0},
                            {5.0, 5.0, 5.0, 5.0, 5.0, 5.0},
                            {6.0, 6.0, 6.0, 6.0, 6.0, 6.0}},
                           {{1.0, 2.0, 3.0, 4.0, 5.0, 6.0},
                            {1.0, 2.0, 3.0, 4.0, 5.0, 6.0},
                            {1.0, 2.0, 3.0, 4.0, 5.0, 6.0},
                            {1.0, 2.0, 3.0, 4.0, 5.0, 6.0},
                            {1.0, 2.0, 3.0, 4.0, 5.0, 6.0},
                            {1.0, 2.0, 3.0, 4.0, 5.0, 6.0}}},
                    "6x6_rect_pattern"),

    std::make_tuple(InType{{{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0},
                            {8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0},
                            {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0},
                            {2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0},
                            {3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0},
                            {4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0},
                            {5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0},
                            {6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0}},
                           {{1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
                            {0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
                            {0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0},
                            {0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0},
                            {0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0},
                            {0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0},
                            {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0},
                            {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0}}},
                    "8x8_identity_right"),

    std::make_tuple(InType{{{0.0, 0.0}, {0.0, 0.0}}, {{1.0, 2.0}, {3.0, 4.0}}}, "2x2_zero_left"),

    std::make_tuple(
        InType{{{1.25, -2.5, 3.75, 0.5}, {0.0, 4.0, -1.0, 2.0}, {3.0, 1.5, 2.25, -3.0}, {2.0, 0.0, 1.0, 1.0}},
               {{2.0, 1.0, 0.0, -1.0}, {3.0, -2.0, 1.0, 0.0}, {0.5, 4.0, -3.0, 2.0}, {1.0, 0.0, 2.0, 3.0}}},
        "4x4_fractions_mix")};

const auto kTestTasksList = std::tuple_cat(ppc::util::AddFuncTask<SannikovIShtrassenAlgorithmMPI, InType>(
                                               kTestParam, PPC_SETTINGS_sannikov_i_shtrassen_algorithm),
                                           ppc::util::AddFuncTask<SannikovIShtrassenAlgorithmSEQ, InType>(
                                               kTestParam, PPC_SETTINGS_sannikov_i_shtrassen_algorithm));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName =
    SannikovIShtrassenAlgorithmFuncTests::PrintFuncTestName<SannikovIShtrassenAlgorithmFuncTests>;

INSTANTIATE_TEST_SUITE_P(PicMatrixTests, SannikovIShtrassenAlgorithmFuncTests, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace sannikov_i_shtrassen_algorithm
