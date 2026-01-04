#include <gtest/gtest.h>
#include <mpi.h>

#include <cmath>
#include <cstddef>
#include <vector>

#include "maslova_u_row_matr_vec_mult/common/include/common.hpp"
#include "maslova_u_row_matr_vec_mult/mpi/include/ops_mpi.hpp"
#include "maslova_u_row_matr_vec_mult/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace maslova_u_row_matr_vec_mult {

class MaslovaURowMatrVecPerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
 private:
  InType input_data_;
  OutType expected_output_;

 protected:
  void SetUp() override {
    const size_t rows = 10000;
    const size_t cols = 10000;

    input_data_.first.rows = rows;
    input_data_.first.cols = cols;
    input_data_.first.data.assign(rows * cols, 1.0);
    input_data_.second.assign(cols, 1.0);

    expected_output_.assign(rows, static_cast<double>(cols));
  }

  InType GetTestInputData() final {
    return input_data_;
  }

  bool CheckTestOutputData(OutType &output_data) final {
    int rank = 0;
    int is_initialized = 0;
    MPI_Initialized(&is_initialized);

    if (is_initialized != 0) {
      MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    }

    if (rank != 0) {
      return true;
    }

    if (output_data.size() != expected_output_.size()) {
      return false;
    }
    for (size_t i = 0; i < output_data.size(); ++i) {
      if (std::abs(output_data[i] - expected_output_[i]) > 1e-6) {
        return false;
      }
    }
    return true;
  }
};

TEST_P(MaslovaURowMatrVecPerfTests, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks = ppc::util::MakeAllPerfTasks<InType, MaslovaURowMatrVecMultMPI, MaslovaURowMatrVecMultSEQ>(
    PPC_SETTINGS_maslova_u_row_matr_vec_mult);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = MaslovaURowMatrVecPerfTests::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, MaslovaURowMatrVecPerfTests, kGtestValues, kPerfTestName);

}  // namespace maslova_u_row_matr_vec_mult
