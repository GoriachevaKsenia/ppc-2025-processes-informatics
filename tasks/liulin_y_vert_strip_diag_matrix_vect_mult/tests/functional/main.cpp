#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <fstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#include "liulin_y_vert_strip_diag_matrix_vect_mult/common/include/common.hpp"
#include "liulin_y_vert_strip_diag_matrix_vect_mult/mpi/include/ops_mpi.hpp"
#include "liulin_y_vert_strip_diag_matrix_vect_mult/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace liulin_y_vert_strip_diag_matrix_vect_mult {

class LiulinYVertStripDiagMatrixVectMultFuncTestsFromFile
    : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &p) {
    return std::get<1>(p);
  }

 protected:
  void SetUp() override {
    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());

    std::string filename = std::get<1>(params);
    std::string abs_path = ppc::util::GetAbsoluteTaskPath(PPC_ID_liulin_y_vert_strip_diag_matrix_vect_mult, filename);

    std::ifstream file(abs_path + ".txt");
    if (!file.is_open()) {
      throw std::runtime_error("Cannot open test file: " + abs_path + ".txt");
    }

    int rows = 0;
    int cols = 0;
    file >> rows >> cols;

    std::vector<std::vector<int>> matrix(rows, std::vector<int>(cols));
    for (int i = 0; i < rows; ++i) {
      for (int j = 0; j < cols; ++j) {
        file >> matrix[i][j];
      }
    }

    std::vector<int> vect(cols);
    for (int j = 0; j < cols; ++j) {
      file >> vect[j];
    }
    if (vect.size() != static_cast<std::size_t>(cols)) {
      throw std::runtime_error("Incompatible shapes: matrix has " + std::to_string(cols) + " columns, but vector has " +
                               std::to_string(vect.size()) + " elements");
    }
    input_data_ = std::make_tuple(matrix, vect);

    exp_output_.assign(rows, 0);
    for (int i = 0; i < rows; ++i) {
      for (int j = 0; j < cols; ++j) {
        exp_output_[i] += matrix[i][j] * vect[j];
      }
    }

    file.close();
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return output_data == exp_output_;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
  OutType exp_output_;
};

namespace {

TEST_P(LiulinYVertStripDiagMatrixVectMultFuncTestsFromFile, MatrixVectorMultFromFile) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 10> kTestParam = {std::make_tuple(0, "tiny"),      std::make_tuple(1, "simple"),
                                             std::make_tuple(2, "square"),    std::make_tuple(3, "rectTall"),
                                             std::make_tuple(4, "rectWide"),  std::make_tuple(5, "singleElement"),
                                             std::make_tuple(6, "singleRow"), std::make_tuple(7, "singleColumn"),
                                             std::make_tuple(8, "negative"),  std::make_tuple(9, "zeroVector")};

const auto kTestTasksList = std::tuple_cat(ppc::util::AddFuncTask<LiulinYVertStripDiagMatrixVectMultMPI, InType>(
                                               kTestParam, PPC_SETTINGS_liulin_y_vert_strip_diag_matrix_vect_mult),
                                           ppc::util::AddFuncTask<LiulinYVertStripDiagMatrixVectMultSEQ, InType>(
                                               kTestParam, PPC_SETTINGS_liulin_y_vert_strip_diag_matrix_vect_mult));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kFuncTestName = LiulinYVertStripDiagMatrixVectMultFuncTestsFromFile::PrintFuncTestName<
    LiulinYVertStripDiagMatrixVectMultFuncTestsFromFile>;

INSTANTIATE_TEST_SUITE_P(FileTests, LiulinYVertStripDiagMatrixVectMultFuncTestsFromFile, kGtestValues, kFuncTestName);

}  // namespace
}  // namespace liulin_y_vert_strip_diag_matrix_vect_mult
