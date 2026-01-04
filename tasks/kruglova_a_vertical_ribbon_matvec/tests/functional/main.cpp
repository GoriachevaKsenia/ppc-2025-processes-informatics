#include <gtest/gtest.h>
#include <mpi.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "kruglova_a_vertical_ribbon_matvec/common/include/common.hpp"
#include "kruglova_a_vertical_ribbon_matvec/mpi/include/ops_mpi.hpp"
#include "kruglova_a_vertical_ribbon_matvec/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace kruglova_a_vertical_ribbon_matvec {

class KruglovaAVerticalRibMatFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &param) {
    return std::to_string(std::get<0>(param)) + "x" + std::to_string(std::get<1>(param));
  }

 protected:
  void SetUp() override {
    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());

    const int rows = std::get<0>(params);
    const int cols = std::get<1>(params);

    const std::size_t matrix_size = static_cast<std::size_t>(rows) * static_cast<std::size_t>(cols);
    std::vector<double> matrix(matrix_size);
    std::vector<double> vec(static_cast<std::size_t>(cols));

    auto n_cols = static_cast<std::size_t>(cols);
    for (std::size_t i = 0; i < n_cols; ++i) {
      matrix[i] = static_cast<double>(i + 1);
    }
    for (std::size_t i = 0; i < n_cols; ++i) {
      vec[i] = static_cast<double>(i + 1);
    }

    ref_output_.resize(static_cast<std::size_t>(rows));
    for (int i = 0; i < rows; ++i) {
      double sum = 0.0;
      for (int j = 0; j < cols; ++j) {
        sum += matrix[(static_cast<std::size_t>(i) * static_cast<std::size_t>(cols)) + static_cast<std::size_t>(j)] *
               vec[static_cast<std::size_t>(j)];
      }
      ref_output_[static_cast<std::size_t>(i)] = sum;
    }

    input_data_ = std::make_tuple(rows, cols, std::move(matrix), std::move(vec));
  }

  bool CheckTestOutputData(OutType &output_data) final {
    int rank = 0;
    int is_mpi_init = 0;

    MPI_Initialized(&is_mpi_init);
    if (is_mpi_init != 0) {
      MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    }

    if (rank != 0) {
      return true;
    }

    if (output_data.size() != ref_output_.size()) {
      return false;
    }

    const double k_epsilon = 1e-6;
    for (std::size_t i = 0; i < output_data.size(); ++i) {
      if (std::abs(output_data[i] - ref_output_[i]) > k_epsilon) {
        return false;
      }
    }
    return true;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
  OutType ref_output_;
};

namespace {

TEST_P(KruglovaAVerticalRibMatFuncTests, MatmulFromPic) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 5> kTestParam = {std::make_tuple(10, 10), std::make_tuple(100, 100),
                                            std::make_tuple(500, 100), std::make_tuple(100, 500),
                                            std::make_tuple(50, 50)};

const auto kTestTasksList = std::tuple_cat(ppc::util::AddFuncTask<KruglovaAVerticalRibbMatMPI, InType>(
                                               kTestParam, PPC_SETTINGS_kruglova_a_vertical_ribbon_matvec),
                                           ppc::util::AddFuncTask<KruglovaAVerticalRibbMatSEQ, InType>(
                                               kTestParam, PPC_SETTINGS_kruglova_a_vertical_ribbon_matvec));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName = KruglovaAVerticalRibMatFuncTests::PrintFuncTestName<KruglovaAVerticalRibMatFuncTests>;

INSTANTIATE_TEST_SUITE_P(PicMatrixTests, KruglovaAVerticalRibMatFuncTests, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace kruglova_a_vertical_ribbon_matvec
