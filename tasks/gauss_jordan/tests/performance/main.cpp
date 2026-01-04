#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <vector>

#include "gauss_jordan/common/include/common.hpp"
#include "gauss_jordan/mpi/include/ops_mpi.hpp"
#include "gauss_jordan/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace gauss_jordan {

// Тест производительности для маленьких матриц (200x200)
class GaivoronskiyMRunPerfTestsSmall : public ppc::util::BaseRunPerfTests<InType, OutType> {
  const int kMatrixSize_ = 200;
  InType input_data_;

  void SetUp() override {
    int n = kMatrixSize_;
    int m = kMatrixSize_;
    input_data_ = InType(static_cast<size_t>(n), std::vector<double>(static_cast<size_t>(m + 1)));

    for (int i = 0; i < n; i++) {
      double sum = 0;
      for (int j = 0; j < m; j++) {
        if (i == j) {
          input_data_[i][j] = 8.0 + ((i % 50) / 5.0);
        } else if (std::abs(i - j) <= 2) {
          input_data_[i][j] = 0.3;
        } else {
          input_data_[i][j] = (i + j) % 7 / 7.0;
        }
        sum += std::abs(input_data_[i][j]);
      }
      input_data_[i][m] = sum;
    }
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return !output_data.empty();
  }

  InType GetTestInputData() final {
    return input_data_;
  }
};

// Тест производительности для средних матриц (500x500)
class GaivoronskiyMRunPerfTestsMedium : public ppc::util::BaseRunPerfTests<InType, OutType> {
  const int kMatrixSize_ = 500;
  InType input_data_;

  void SetUp() override {
    int n = kMatrixSize_;
    int m = kMatrixSize_;
    input_data_ = InType(static_cast<size_t>(n), std::vector<double>(static_cast<size_t>(m + 1)));

    for (int i = 0; i < n; i++) {
      double sum = 0;
      for (int j = 0; j < m; j++) {
        if (i == j) {
          input_data_[i][j] = 15.0 + ((i % 100) / 8.0);
        } else if (std::abs(i - j) <= 3) {
          input_data_[i][j] = 0.4;
        } else {
          input_data_[i][j] = ((i * j) % 11) / 11.0;
        }
        sum += std::abs(input_data_[i][j]);
      }
      input_data_[i][m] = sum;
    }
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return !output_data.empty();
  }

  InType GetTestInputData() final {
    return input_data_;
  }
};

// Тест производительности для больших матриц (1000x1000)
class GaivoronskiyMRunPerfTestsLarge : public ppc::util::BaseRunPerfTests<InType, OutType> {
  const int kMatrixSize_ = 1000;
  InType input_data_;

  void SetUp() override {
    int n = kMatrixSize_;
    int m = kMatrixSize_;
    input_data_ = InType(static_cast<size_t>(n), std::vector<double>(static_cast<size_t>(m + 1)));

    for (int i = 0; i < n; i++) {
      double sum = 0;
      for (int j = 0; j < m; j++) {
        if (i == j) {
          input_data_[i][j] = 25.0 + ((i % 150) / 10.0);
        } else if (std::abs(i - j) <= 5) {
          input_data_[i][j] = 0.2;
        } else {
          input_data_[i][j] = ((i + j * 3) % 13) / 13.0;
        }
        sum += std::abs(input_data_[i][j]);
      }
      input_data_[i][m] = sum;
    }
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return !output_data.empty();
  }

  InType GetTestInputData() final {
    return input_data_;
  }
};

TEST_P(GaivoronskiyMRunPerfTestsSmall, RunPerfModesSmall) {
  ExecuteTest(GetParam());
}

TEST_P(GaivoronskiyMRunPerfTestsMedium, RunPerfModesMedium) {
  ExecuteTest(GetParam());
}

TEST_P(GaivoronskiyMRunPerfTestsLarge, RunPerfModesLarge) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasksSmall =
    ppc::util::MakeAllPerfTasks<InType, GaussJordanMPI, GaussJordanSEQ>(PPC_SETTINGS_gauss_jordan);

const auto kAllPerfTasksMedium =
    ppc::util::MakeAllPerfTasks<InType, GaussJordanMPI, GaussJordanSEQ>(PPC_SETTINGS_gauss_jordan);

const auto kAllPerfTasksLarge =
    ppc::util::MakeAllPerfTasks<InType, GaussJordanMPI, GaussJordanSEQ>(PPC_SETTINGS_gauss_jordan);

const auto kGtestValuesSmall = ppc::util::TupleToGTestValues(kAllPerfTasksSmall);
const auto kGtestValuesMedium = ppc::util::TupleToGTestValues(kAllPerfTasksMedium);
const auto kGtestValuesLarge = ppc::util::TupleToGTestValues(kAllPerfTasksLarge);

const auto kPerfTestNameSmall = GaivoronskiyMRunPerfTestsSmall::CustomPerfTestName;
const auto kPerfTestNameMedium = GaivoronskiyMRunPerfTestsMedium::CustomPerfTestName;
const auto kPerfTestNameLarge = GaivoronskiyMRunPerfTestsLarge::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTestsSmall, GaivoronskiyMRunPerfTestsSmall, kGtestValuesSmall, kPerfTestNameSmall);
INSTANTIATE_TEST_SUITE_P(RunModeTestsMedium, GaivoronskiyMRunPerfTestsMedium, kGtestValuesMedium, kPerfTestNameMedium);
INSTANTIATE_TEST_SUITE_P(RunModeTestsLarge, GaivoronskiyMRunPerfTestsLarge, kGtestValuesLarge, kPerfTestNameLarge);

}  // namespace gauss_jordan
