#include <gtest/gtest.h>

#include <fstream>
#include <stdexcept>
#include <string>

#include "savva_d_min_elem_vec//common/include/common.hpp"
#include "savva_d_min_elem_vec/mpi/include/ops_mpi.hpp"
#include "savva_d_min_elem_vec/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"
#include "util/include/util.hpp"

namespace savva_d_min_elem_vec {

class SavvaDMinElemVecPerfTest : public ppc::util::BaseRunPerfTests<InType, OutType> {
  InType input_data_;

  int expected_min_ = 0;

  void SetUp() override {
    // Чтение данных из файла

    std::string file_path = ppc::util::GetAbsoluteTaskPath(PPC_ID_savva_d_min_elem_vec, "data.txt");
    std::ifstream file(file_path);

    if (!file.is_open()) {
      throw std::runtime_error("Cannot open test data file");
    }

    // Читаем размер вектора
    int vector_size = 0;
    file >> vector_size;

    // Читаем ожидаемый минимум
    file >> expected_min_;

    // Читаем данные вектора
    input_data_.resize(vector_size + 9000000);
    for (int i = 0; i < 9000000; ++i) {
      file >> input_data_[i];
    }

    for (int i = 0; i < 9000000; ++i) {
      input_data_[9000000 + i] = input_data_[i] - 9;
    }

    for (int i = 9000000; i < vector_size; ++i) {
      file >> input_data_[i];
    }

    file.close();

    // Проверяем что данные загружены корректно
    if (input_data_.empty()) {
      throw std::runtime_error("Test data is empty!");
    }
  }

  // случайная генерация в SetUp будет блокироваться, так как производительность зависит от сгенерированных данных
  // данные для perfomance должны быть сгенерированы в data,упакованы, распакованы в SetUp()

  bool CheckTestOutputData(OutType &output_data) final {
    return expected_min_ == output_data;
  }

  InType GetTestInputData() final {
    return input_data_;
  }
};

TEST_P(SavvaDMinElemVecPerfTest, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, SavvaDMinElemVecMPI, SavvaDMinElemVecSEQ>(PPC_SETTINGS_savva_d_min_elem_vec);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = SavvaDMinElemVecPerfTest::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, SavvaDMinElemVecPerfTest, kGtestValues, kPerfTestName);

}  // namespace savva_d_min_elem_vec
