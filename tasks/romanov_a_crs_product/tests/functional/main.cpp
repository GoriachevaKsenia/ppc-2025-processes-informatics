#include <gtest/gtest.h>
#include <stb/stb_image.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <initializer_list>
#include <string>
#include <tuple>

#include "romanov_a_crs_product/common/include/common.hpp"
#include "romanov_a_crs_product/mpi/include/ops_mpi.hpp"
#include "romanov_a_crs_product/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace romanov_a_crs_product {

class RomanovACRSProductFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    const auto &[a, b, c] = test_param;
    return "_n" + std::to_string(a.GetRows()) + "_nnzA" + std::to_string(a.Nnz()) + "_nnzB" + std::to_string(b.Nnz());
  }

 protected:
  void SetUp() override {
    TestType params = std::get<static_cast<std::uint64_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());

    const auto &[a, b, c] = params;
    input_data_ = std::make_tuple(a, b);
    expected_ = c;
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return (output_data == expected_);
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
  OutType expected_;
};

namespace {

TEST_P(RomanovACRSProductFuncTests, MatmulFromPic) {
  ExecuteTest(GetParam());
}

CRS MakeDenseCRS(std::initializer_list<std::initializer_list<double>> rows) {
  uint64_t n = rows.size();
  Dense d(n);
  uint64_t i = 0;
  for (const auto &r : rows) {
    uint64_t j = 0;
    for (const auto &v : r) {
      d(i, j) = v;
      ++j;
    }
    ++i;
  }
  return ToCRS(d);
}

const std::array<TestType, 5> kTestParam = {
    TestType{
        MakeDenseCRS({{2.0}}),
        MakeDenseCRS({{3.0}}),
        MakeDenseCRS({{6.0}}),
    },
    TestType{
        MakeDenseCRS({{1.0, 0.0}, {0.0, 1.0}}),
        MakeDenseCRS({{5.0, 6.0}, {7.0, 8.0}}),
        MakeDenseCRS({{5.0, 6.0}, {7.0, 8.0}}),
    },
    TestType{
        MakeDenseCRS({{1.0, 0.0, 2.0}, {0.0, 0.0, 0.0}, {3.0, 0.0, 4.0}}),
        MakeDenseCRS({{0.0, 5.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 6.0, 0.0}}),
        MakeDenseCRS({{0.0, 17.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 39.0, 0.0}}),
    },
    TestType{
        MakeDenseCRS({{0.0, 2.0, 0.0, 1.0}, {0.0, 0.0, 0.0, 0.0}, {0.0, 0.0, 3.0, 0.0}, {4.0, 0.0, 0.0, 0.0}}),
        MakeDenseCRS({{0.0, 5.0, 0.0, 0.0}, {7.0, 0.0, 0.0, 0.0}, {0.0, 0.0, 6.0, 0.0}, {0.0, 0.0, 0.0, 8.0}}),
        MakeDenseCRS({{14.0, 0.0, 0.0, 8.0}, {0.0, 0.0, 0.0, 0.0}, {0.0, 0.0, 18.0, 0.0}, {0.0, 20.0, 0.0, 0.0}}),
    },
    TestType{
        MakeDenseCRS({{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}, {7.0, 8.0, 9.0}}),
        MakeDenseCRS({{1.0, 4.0, 7.0}, {2.0, 5.0, 8.0}, {3.0, 6.0, 9.0}}),
        MakeDenseCRS({{14.0, 32.0, 50.0}, {32.0, 77.0, 122.0}, {50.0, 122.0, 194.0}}),
    },
};

const auto kTestTasksList = std::tuple_cat(
    ppc::util::AddFuncTask<RomanovACRSProductMPI, InType>(kTestParam, PPC_SETTINGS_romanov_a_crs_product),
    ppc::util::AddFuncTask<RomanovACRSProductSEQ, InType>(kTestParam, PPC_SETTINGS_romanov_a_crs_product));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName = RomanovACRSProductFuncTests::PrintFuncTestName<RomanovACRSProductFuncTests>;

INSTANTIATE_TEST_SUITE_P(PicMatrixTests, RomanovACRSProductFuncTests, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace romanov_a_crs_product
