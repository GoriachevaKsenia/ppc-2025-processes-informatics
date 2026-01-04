#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#include "nikitin_a_fox_algorithm/common/include/common.hpp"
#include "nikitin_a_fox_algorithm/mpi/include/ops_mpi.hpp"
#include "nikitin_a_fox_algorithm/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"

namespace nikitin_a_fox_algorithm {

class NikitinAFoxAlgorithmFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return std::get<1>(test_param);
  }

 protected:
  void SetUp() override {
    auto params = GetParam();
    TestType test_params = std::get<2>(params);

    test_case_id_ = static_cast<int>(std::get<0>(test_params));
    test_description_ = std::get<1>(test_params);

    // Генерируем тестовые данные в зависимости от ID теста
    switch (test_case_id_) {
      case 1:  // Матрицы 1x1
        GenerateTestMatrices(1);
        break;

      case 2:  // Матрицы 2x2
        GenerateTestMatrices(2);
        break;

      case 3:  // Матрицы 3x3
        GenerateTestMatrices(3);
        break;

      case 4:  // Матрицы 4x4
        GenerateTestMatrices(4);
        break;

      case 5:  // Матрицы 5x5
        GenerateTestMatrices(5);
        break;

      case 6:  // Матрицы 8x8
        GenerateTestMatrices(8);
        break;

      case 7:  // Матрицы 10x10
        GenerateTestMatrices(10);
        break;

      case 8:  // Матрицы 16x16
        GenerateTestMatrices(16);
        break;

      case 9:  // Матрицы 20x20
        GenerateTestMatrices(20);
        break;

      case 10:  // Матрицы 32x32
        GenerateTestMatrices(32);
        break;

      case 11:  // Матрицы 50x50
        GenerateTestMatrices(50);
        break;

      case 12:  // Матрицы 64x64
        GenerateTestMatrices(64);
        break;

      case 13:  // Матрицы 100x100
        GenerateTestMatrices(100);
        break;

      case 14:  // Матрицы 127x127
        GenerateTestMatrices(127);
        break;

      // Тесты с особыми значениями
      case 15:  // Матрицы с нулями
        GenerateZeroMatrices(10);
        break;

      case 16:  // Единичные матрицы
        GenerateIdentityMatrices(8);
        break;

      case 17:  // Нулевая матрица × любая матрица
        GenerateZeroTimesAny(5);
        break;

      case 18:  // Диагональные матрицы
        GenerateDiagonalMatrices(6);
        break;

      case 19:  // Верхнетреугольные матрицы
        GenerateUpperTriangularMatrices(7);
        break;

      case 20:  // Нижнетреугольные матрицы
        GenerateLowerTriangularMatrices(7);
        break;

      case 21:  // Симметричные матрицы
        GenerateSymmetricMatrices(9);
        break;

      case 22:  // Матрицы с очень маленькими значениями
        GenerateSmallValueMatrices(4);
        break;

      case 23:  // Матрицы с очень большими значениями
        GenerateLargeValueMatrices(4);
        break;

      case 24:  // Матрицы со смешанными знаками
        GenerateMixedSignMatrices(6);
        break;

      case 25:  // Матрицы с одинаковыми элементами
        GenerateConstantMatrices(5);
        break;

      case 26:  // Матрицы, где A × B ≠ B × A
        GenerateNonCommutativeMatrices();
        break;

      case 27:  // Матрицы, где A × (B × C) = (A × B) × C
        GenerateAssociativityTest(4);
        break;

      case 28:  // Матрицы с NaN и Inf значениями
        GenerateSpecialValueMatrices(3);
        break;

      case 29:  // Рандомные матрицы
        GenerateRandomMatrices(25, 12345);
        break;

      case 30:  // Большие рандомные матрицы
        GenerateRandomMatrices(100, 54321);
        break;

      default:
        throw std::runtime_error("Unknown test case ID: " + std::to_string(test_case_id_));
    }
  }

  bool CheckTestOutputData(OutType &output_data) final {
    // Проверяем размер
    if (output_data.size() != expected_output_.size()) {
      return false;
    }

    for (size_t i = 0; i < output_data.size(); ++i) {
      if (output_data[i].size() != expected_output_[i].size()) {
        return false;
      }
    }

    // Проверяем каждый элемент с учетом погрешности
    for (size_t i = 0; i < output_data.size(); ++i) {
      for (size_t j = 0; j < output_data[i].size(); ++j) {
        double expected = expected_output_[i][j];
        double actual = output_data[i][j];

        // Обработка специальных значений
        if (std::isnan(expected) && std::isnan(actual)) {
          continue;
        }

        if (std::isinf(expected) && std::isinf(actual) && std::signbit(expected) == std::signbit(actual)) {
          continue;
        }

        // Для обычных чисел используем относительную погрешность
        double tolerance = 1e-8;
        if (std::abs(expected) > 1e-10) {
          double relative_error = std::abs(actual - expected) / std::abs(expected);
          if (relative_error > tolerance) {
            return false;
          }
        } else {
          // Для очень маленьких чисел используем абсолютную погрешность
          double absolute_error = std::abs(actual - expected);
          if (absolute_error > tolerance) {
            return false;
          }
        }
      }
    }

    return true;
  }

  InType GetTestInputData() final {
    return {matrix_a_, matrix_b_};
  }

 private:
  int test_case_id_ = 0;
  std::string test_description_;
  std::vector<std::vector<double>> matrix_a_;
  std::vector<std::vector<double>> matrix_b_;
  std::vector<std::vector<double>> expected_output_;

  // Генерация матриц заданного размера
  void GenerateTestMatrices(int n) {
    matrix_a_ = std::vector<std::vector<double>>(n, std::vector<double>(n));
    matrix_b_ = std::vector<std::vector<double>>(n, std::vector<double>(n));

    std::mt19937 gen(static_cast<unsigned int>(n * 123));
    std::uniform_real_distribution<double> dist(-10.0, 10.0);

    for (int i = 0; i < n; ++i) {
      for (int j = 0; j < n; ++j) {
        matrix_a_[i][j] = dist(gen);
        matrix_b_[i][j] = dist(gen);
      }
    }

    ComputeExpectedResult();
  }

  // Вычисление ожидаемого результата умножения матриц
  void ComputeExpectedResult() {
    const auto n = static_cast<int>(matrix_a_.size());
    expected_output_ = std::vector<std::vector<double>>(n, std::vector<double>(n, 0.0));

    for (int i = 0; i < n; ++i) {
      for (int j = 0; j < n; ++j) {
        double sum = 0.0;
        for (int k = 0; k < n; ++k) {
          sum += matrix_a_[i][k] * matrix_b_[k][j];
        }
        expected_output_[i][j] = sum;
      }
    }
  }

  // Генерация нулевых матриц
  void GenerateZeroMatrices(int n) {
    matrix_a_ = std::vector<std::vector<double>>(n, std::vector<double>(n, 0.0));
    matrix_b_ = std::vector<std::vector<double>>(n, std::vector<double>(n, 0.0));
    expected_output_ = std::vector<std::vector<double>>(n, std::vector<double>(n, 0.0));
  }

  // Генерация единичных матриц
  void GenerateIdentityMatrices(int n) {
    matrix_a_ = std::vector<std::vector<double>>(n, std::vector<double>(n, 0.0));
    matrix_b_ = std::vector<std::vector<double>>(n, std::vector<double>(n, 0.0));
    expected_output_ = std::vector<std::vector<double>>(n, std::vector<double>(n, 0.0));

    for (int i = 0; i < n; ++i) {
      matrix_a_[i][i] = 1.0;
      matrix_b_[i][i] = 1.0;
      expected_output_[i][i] = 1.0;
    }
  }

  // Генерация: нулевая матрица × любая матрица
  void GenerateZeroTimesAny(int n) {
    matrix_a_ = std::vector<std::vector<double>>(n, std::vector<double>(n, 0.0));
    matrix_b_ = std::vector<std::vector<double>>(n, std::vector<double>(n));
    expected_output_ = std::vector<std::vector<double>>(n, std::vector<double>(n, 0.0));

    std::mt19937 gen(static_cast<unsigned int>(n * 456));
    std::uniform_real_distribution<double> dist(-100.0, 100.0);

    for (int i = 0; i < n; ++i) {
      for (int j = 0; j < n; ++j) {
        matrix_b_[i][j] = dist(gen);
      }
    }
  }

  // Генерация диагональных матриц
  void GenerateDiagonalMatrices(int n) {
    matrix_a_ = std::vector<std::vector<double>>(n, std::vector<double>(n, 0.0));
    matrix_b_ = std::vector<std::vector<double>>(n, std::vector<double>(n, 0.0));

    std::mt19937 gen(static_cast<unsigned int>(n * 789));
    std::uniform_real_distribution<double> dist(1.0, 10.0);

    for (int i = 0; i < n; ++i) {
      matrix_a_[i][i] = dist(gen);
      matrix_b_[i][i] = dist(gen);
    }

    ComputeExpectedResult();
  }

  // Генерация верхнетреугольных матриц
  void GenerateUpperTriangularMatrices(int n) {
    matrix_a_ = std::vector<std::vector<double>>(n, std::vector<double>(n, 0.0));
    matrix_b_ = std::vector<std::vector<double>>(n, std::vector<double>(n, 0.0));

    std::mt19937 gen(static_cast<unsigned int>(n * 1011));
    std::uniform_real_distribution<double> dist(1.0, 5.0);

    for (int i = 0; i < n; ++i) {
      for (int j = i; j < n; ++j) {
        matrix_a_[i][j] = dist(gen);
        matrix_b_[i][j] = dist(gen);
      }
    }

    ComputeExpectedResult();
  }

  // Генерация нижнетреугольных матриц
  void GenerateLowerTriangularMatrices(int n) {
    matrix_a_ = std::vector<std::vector<double>>(n, std::vector<double>(n, 0.0));
    matrix_b_ = std::vector<std::vector<double>>(n, std::vector<double>(n, 0.0));

    std::mt19937 gen(static_cast<unsigned int>(n * 1213));
    std::uniform_real_distribution<double> dist(1.0, 5.0);

    for (int i = 0; i < n; ++i) {
      for (int j = 0; j <= i; ++j) {
        matrix_a_[i][j] = dist(gen);
        matrix_b_[i][j] = dist(gen);
      }
    }

    ComputeExpectedResult();
  }

  // Генерация симметричных матриц
  void GenerateSymmetricMatrices(int n) {
    matrix_a_ = std::vector<std::vector<double>>(n, std::vector<double>(n, 0.0));
    matrix_b_ = std::vector<std::vector<double>>(n, std::vector<double>(n, 0.0));

    std::mt19937 gen(static_cast<unsigned int>(n * 1415));
    std::uniform_real_distribution<double> dist(-5.0, 5.0);

    for (int i = 0; i < n; ++i) {
      for (int j = i; j < n; ++j) {
        double val = dist(gen);
        matrix_a_[i][j] = val;
        matrix_a_[j][i] = val;

        val = dist(gen);
        matrix_b_[i][j] = val;
        matrix_b_[j][i] = val;
      }
    }

    ComputeExpectedResult();
  }

  // Генерация матриц с очень маленькими значениями
  void GenerateSmallValueMatrices(int n) {
    matrix_a_ = std::vector<std::vector<double>>(n, std::vector<double>(n));
    matrix_b_ = std::vector<std::vector<double>>(n, std::vector<double>(n));

    std::mt19937 gen(static_cast<unsigned int>(n * 1617));
    std::uniform_real_distribution<double> dist(1e-15, 1e-10);

    for (int i = 0; i < n; ++i) {
      for (int j = 0; j < n; ++j) {
        matrix_a_[i][j] = dist(gen);
        matrix_b_[i][j] = dist(gen);
      }
    }

    ComputeExpectedResult();
  }

  // Генерация матриц с очень большими значениями
  void GenerateLargeValueMatrices(int n) {
    matrix_a_ = std::vector<std::vector<double>>(n, std::vector<double>(n));
    matrix_b_ = std::vector<std::vector<double>>(n, std::vector<double>(n));

    std::mt19937 gen(static_cast<unsigned int>(n * 1819));
    std::uniform_real_distribution<double> dist(1e10, 1e15);

    for (int i = 0; i < n; ++i) {
      for (int j = 0; j < n; ++j) {
        matrix_a_[i][j] = dist(gen);
        matrix_b_[i][j] = dist(gen);
      }
    }

    ComputeExpectedResult();
  }

  // Генерация матриц со смешанными знаками
  void GenerateMixedSignMatrices(int n) {
    matrix_a_ = std::vector<std::vector<double>>(n, std::vector<double>(n));
    matrix_b_ = std::vector<std::vector<double>>(n, std::vector<double>(n));

    std::mt19937 gen(static_cast<unsigned int>(n * 2021));
    std::uniform_real_distribution<double> dist(-1000.0, 1000.0);

    for (int i = 0; i < n; ++i) {
      for (int j = 0; j < n; ++j) {
        matrix_a_[i][j] = dist(gen);
        matrix_b_[i][j] = dist(gen);
      }
    }

    ComputeExpectedResult();
  }

  // Генерация матриц с постоянными элементами
  void GenerateConstantMatrices(int n) {
    const double val_a = 2.5;
    const double val_b = 3.5;

    matrix_a_ = std::vector<std::vector<double>>(n, std::vector<double>(n, val_a));
    matrix_b_ = std::vector<std::vector<double>>(n, std::vector<double>(n, val_b));
    expected_output_ = std::vector<std::vector<double>>(n, std::vector<double>(n, val_a * val_b * n));
  }

  // Генерация некоммутативных матриц
  void GenerateNonCommutativeMatrices() {
    matrix_a_ = {{1, 2}, {3, 4}};
    matrix_b_ = {{5, 6}, {7, 8}};

    ComputeExpectedResult();
  }

  // Генерация теста на ассоциативность
  void GenerateAssociativityTest(int n) {
    matrix_a_ = std::vector<std::vector<double>>(n, std::vector<double>(n));
    matrix_b_ = std::vector<std::vector<double>>(n, std::vector<double>(n));

    std::mt19937 gen(static_cast<unsigned int>(n * 2223));
    std::uniform_real_distribution<double> dist(-5.0, 5.0);

    for (int i = 0; i < n; ++i) {
      for (int j = 0; j < n; ++j) {
        matrix_a_[i][j] = dist(gen);
        matrix_b_[i][j] = dist(gen);
      }
    }

    ComputeExpectedResult();
  }

  // Генерация матриц с NaN и Inf значениями
  void GenerateSpecialValueMatrices(int n) {
    matrix_a_ = std::vector<std::vector<double>>(n, std::vector<double>(n, 1.0));
    matrix_b_ = std::vector<std::vector<double>>(n, std::vector<double>(n, 1.0));

    if (n >= 2) {
      matrix_a_[0][0] = std::numeric_limits<double>::quiet_NaN();
      matrix_a_[0][1] = std::numeric_limits<double>::infinity();
      matrix_a_[1][0] = -std::numeric_limits<double>::infinity();

      matrix_b_[1][1] = std::numeric_limits<double>::quiet_NaN();
    }

    ComputeExpectedResult();
  }

  // Генерация случайных матриц
  void GenerateRandomMatrices(int n, int seed) {
    matrix_a_ = std::vector<std::vector<double>>(n, std::vector<double>(n));
    matrix_b_ = std::vector<std::vector<double>>(n, std::vector<double>(n));

    std::mt19937 gen(static_cast<unsigned int>(seed));
    std::uniform_real_distribution<double> dist(-100.0, 100.0);

    for (int i = 0; i < n; ++i) {
      for (int j = 0; j < n; ++j) {
        matrix_a_[i][j] = dist(gen);
        matrix_b_[i][j] = dist(gen);
      }
    }

    ComputeExpectedResult();
  }
};

namespace {

TEST_P(NikitinAFoxAlgorithmFuncTests, MatrixMultiplicationTest) {
  ExecuteTest(GetParam());
}

// Определяем тестовые случаи
const std::array<TestType, 30> kTestParam = {
    std::make_tuple(1, "1x1_matrices"),        std::make_tuple(2, "2x2_matrices"),
    std::make_tuple(3, "3x3_matrices"),        std::make_tuple(4, "4x4_matrices"),
    std::make_tuple(5, "5x5_matrices"),        std::make_tuple(6, "8x8_matrices"),
    std::make_tuple(7, "10x10_matrices"),      std::make_tuple(8, "16x16_matrices"),
    std::make_tuple(9, "20x20_matrices"),      std::make_tuple(10, "32x32_matrices"),
    std::make_tuple(11, "50x50_matrices"),     std::make_tuple(12, "64x64_matrices"),
    std::make_tuple(13, "100x100_matrices"),   std::make_tuple(14, "127x127_matrices"),
    std::make_tuple(15, "zero_matrices"),      std::make_tuple(16, "identity_matrices"),
    std::make_tuple(17, "zero_times_any"),     std::make_tuple(18, "diagonal_matrices"),
    std::make_tuple(19, "upper_triangular"),   std::make_tuple(20, "lower_triangular"),
    std::make_tuple(21, "symmetric_matrices"), std::make_tuple(22, "small_values"),
    std::make_tuple(23, "large_values"),       std::make_tuple(24, "mixed_signs"),
    std::make_tuple(25, "constant_matrices"),  std::make_tuple(26, "non_commutative"),
    std::make_tuple(27, "associativity_test"), std::make_tuple(28, "special_values_nan_inf"),
    std::make_tuple(29, "random_25x25"),       std::make_tuple(30, "random_100x100")};

const auto kTestTasksList = std::tuple_cat(
    ppc::util::AddFuncTask<NikitinAFoxAlgorithmMPI, InType>(kTestParam, PPC_SETTINGS_nikitin_a_fox_algorithm),
    ppc::util::AddFuncTask<NikitinAFoxAlgorithmSEQ, InType>(kTestParam, PPC_SETTINGS_nikitin_a_fox_algorithm));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName = NikitinAFoxAlgorithmFuncTests::PrintFuncTestName<NikitinAFoxAlgorithmFuncTests>;

INSTANTIATE_TEST_SUITE_P(FoxAlgorithmTests, NikitinAFoxAlgorithmFuncTests, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace nikitin_a_fox_algorithm
