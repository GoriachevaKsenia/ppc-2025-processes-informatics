#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <random>
#include <vector>

#include "nikitin_a_fox_algorithm/common/include/common.hpp"
#include "nikitin_a_fox_algorithm/mpi/include/ops_mpi.hpp"
#include "nikitin_a_fox_algorithm/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace nikitin_a_fox_algorithm {

class NikitinAFoxAlgorithmPerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
  InType input_data_;

  // Для тестирования производительности умножения матриц используем большие размеры
  // Для MPI особенно важны размеры, которые хорошо делятся на блоки
  const int kMatrixSize_ = 512;  // 512x512 - большой, но реалистичный размер

  void SetUp() override {
    // Генерируем две большие матрицы со случайными данными
    std::vector<std::vector<double>> matrix_a(kMatrixSize_, std::vector<double>(kMatrixSize_));
    std::vector<std::vector<double>> matrix_b(kMatrixSize_, std::vector<double>(kMatrixSize_));

    // Инициализируем генератор случайных чисел
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dist(-100.0, 100.0);

    // Заполняем матрицы случайными значениями
    // Это обеспечит разнообразные вычисления и хорошую нагрузку
    for (int i = 0; i < kMatrixSize_; ++i) {
      for (int j = 0; j < kMatrixSize_; ++j) {
        matrix_a[i][j] = dist(gen);
        matrix_b[i][j] = dist(gen);
      }
    }

    input_data_ = {matrix_a, matrix_b};
  }

  bool CheckTestOutputData(OutType &output_data) final {
    // Проверяем только базовую корректность:
    // 1. Результат не должен быть пустым
    // 2. Должен быть правильного размера
    if (output_data.empty()) {
      return false;
    }

    if (output_data.size() != static_cast<size_t>(kMatrixSize_)) {
      return false;
    }

    for (const auto &row : output_data) {
      if (row.size() != static_cast<size_t>(kMatrixSize_)) {
        return false;
      }
    }

    // Дополнительная проверка: убедимся, что результат содержит разумные значения
    // (не все нули или NaN)
    bool has_valid_values = false;
    int valid_count = 0;

    // Проверяем только небольшую выборку для скорости
    const int sample_size = std::min(100, kMatrixSize_ * kMatrixSize_ / 100);

    for (int i = 0; i < sample_size; ++i) {
      int row = (i * 17) % kMatrixSize_;  // Простая псевдослучайная выборка
      int col = (i * 13) % kMatrixSize_;

      double val = output_data[row][col];
      if (!std::isnan(val) && !std::isinf(val)) {
        valid_count++;
        if (std::abs(val) > 1e-10) {
          has_valid_values = true;
        }
      }
    }

    // Должно быть достаточно много валидных значений
    if (valid_count < sample_size / 2) {
      return false;
    }

    return has_valid_values;
  }

  InType GetTestInputData() final {
    return input_data_;
  }
};

TEST_P(NikitinAFoxAlgorithmPerfTests, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks = ppc::util::MakeAllPerfTasks<InType, NikitinAFoxAlgorithmMPI, NikitinAFoxAlgorithmSEQ>(
    PPC_SETTINGS_nikitin_a_fox_algorithm);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = NikitinAFoxAlgorithmPerfTests::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, NikitinAFoxAlgorithmPerfTests, kGtestValues, kPerfTestName);

}  // namespace nikitin_a_fox_algorithm
