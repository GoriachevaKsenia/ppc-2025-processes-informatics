#include <gtest/gtest.h>

#include <cstddef>
#include <fstream>
#include <string>
#include <vector>

#include "leonova_a_star/common/include/common.hpp"
#include "leonova_a_star/mpi/include/ops_mpi.hpp"
#include "leonova_a_star/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace leonova_a_star {

class LeonovaAStarRunPerfTestsProcesses : public ppc::util::BaseRunPerfTests<InType, OutType> {
  std::vector<InType> test_inputs_;  // Вектор пар матриц для тестирования
  size_t current_input_index_ = 0;   // Индекс текущего теста

  void SetUp() override {
    test_inputs_ = ReadMatricesFromFile("tasks/leonova_a_star/data/matrices_perf.txt");
    current_input_index_ = 0;
  }

  bool CheckTestOutputData(OutType &output_data) final {
    // Для тестов производительности просто проверяем, что результат вычислен
    return !output_data.empty();
  }

  InType GetTestInputData() final {
    if (current_input_index_ < test_inputs_.size()) {
      return test_inputs_[current_input_index_++];
    }

    return test_inputs_.back();
  }

 private:
  static std::vector<InType> ReadMatricesFromFile(const std::string &filepath) {
    std::vector<InType> result;

    std::ifstream file(filepath);
    if (!file.is_open()) {
      return result;
    }

    int num_tests = 0;
    file >> num_tests;

    for (int tndex = 0; tndex < num_tests; tndex++) {
      int rows_a = 0;
      int cols_a = 0;
      file >> rows_a >> cols_a;

      std::vector<std::vector<int>> matrix_a(rows_a, std::vector<int>(cols_a));
      for (int index = 0; index < rows_a; index++) {
        for (int jndex = 0; jndex < cols_a; jndex++) {
          file >> matrix_a[index][jndex];
        }
      }

      int rows_b = 0;
      int cols_b = 0;
      file >> rows_b >> cols_b;

      std::vector<std::vector<int>> matrix_b(rows_b, std::vector<int>(cols_b));
      for (int index = 0; index < rows_b; index++) {
        for (int jndex = 0; jndex < cols_b; jndex++) {
          file >> matrix_b[index][jndex];
        }
      }

      result.emplace_back(matrix_a, matrix_b);
    }

    file.close();
    return result;
  }
};

TEST_P(LeonovaAStarRunPerfTestsProcesses, RunPerfTests) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, LeonovaAStarMPI, LeonovaAStarSEQ>(PPC_SETTINGS_leonova_a_star);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = LeonovaAStarRunPerfTestsProcesses::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunMatrixPerf, LeonovaAStarRunPerfTestsProcesses, kGtestValues, kPerfTestName);

}  // namespace leonova_a_star
