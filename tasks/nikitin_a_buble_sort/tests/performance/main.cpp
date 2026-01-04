#include <gtest/gtest.h>

#include <cstddef>
#include <random>
#include <vector>

#include "nikitin_a_buble_sort/common/include/common.hpp"
#include "nikitin_a_buble_sort/mpi/include/ops_mpi.hpp"
#include "nikitin_a_buble_sort/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace nikitin_a_buble_sort {

// Тест 1: Случайные данные
class NikitinABubleSortRandomPerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
 public:
  void SetUp() override {
    const std::size_t n = 10000;
    std::vector<double> data(n);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dist(-10000.0, 10000.0);

    for (std::size_t i = 0; i < n; ++i) {
      data[i] = dist(gen);
    }

    input_data_ = data;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

  bool CheckTestOutputData(OutType &output_data) final {
    // Для perf-тестов проверяем только что массив отсортирован
    if (output_data.empty()) {
      return false;
    }

    for (size_t i = 1; i < output_data.size(); ++i) {
      if (output_data[i - 1] > output_data[i]) {
        return false;
      }
    }
    return true;
  }

 private:
  InType input_data_;
};

// Тест 2: Данные по возрастанию
class NikitinABubleSortAscendingPerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
 public:
  void SetUp() override {
    const std::size_t n = 10000;
    std::vector<double> data(n);

    for (std::size_t i = 0; i < n; ++i) {
      data[i] = static_cast<double>(i + 1);
    }

    input_data_ = data;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

  bool CheckTestOutputData(OutType &output_data) final {
    // Для perf-тестов проверяем только что массив отсортирован
    if (output_data.empty()) {
      return false;
    }

    for (size_t i = 1; i < output_data.size(); ++i) {
      if (output_data[i - 1] > output_data[i]) {
        return false;
      }
    }
    return true;
  }

 private:
  InType input_data_;
};

// Тест 3: Данные по убыванию
class NikitinABubleSortDescendingPerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
 public:
  void SetUp() override {
    const std::size_t n = 10000;
    std::vector<double> data(n);

    for (std::size_t i = 0; i < n; ++i) {
      data[i] = static_cast<double>(n - i);
    }

    input_data_ = data;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

  bool CheckTestOutputData(OutType &output_data) final {
    // Для perf-тестов проверяем только что массив отсортирован
    if (output_data.empty()) {
      return false;
    }

    for (size_t i = 1; i < output_data.size(); ++i) {
      if (output_data[i - 1] > output_data[i]) {
        return false;
      }
    }
    return true;
  }

 private:
  InType input_data_;
};

namespace {

// Тесты со случайными данными
TEST_P(NikitinABubleSortRandomPerfTests, RunPerfModesRandom) {
  ExecuteTest(GetParam());
}

const auto kRandomPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, NikitinABubleSortMPI, NikitinABubleSortSEQ>(PPC_SETTINGS_nikitin_a_buble_sort);

const auto kRandomGtestValues = ppc::util::TupleToGTestValues(kRandomPerfTasks);

const auto kRandomPerfTestName = NikitinABubleSortRandomPerfTests::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunPerfRandom, NikitinABubleSortRandomPerfTests, kRandomGtestValues, kRandomPerfTestName);

// Тесты с данными по возрастанию
TEST_P(NikitinABubleSortAscendingPerfTests, RunPerfModesAscending) {
  ExecuteTest(GetParam());
}

const auto kAscendingPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, NikitinABubleSortMPI, NikitinABubleSortSEQ>(PPC_SETTINGS_nikitin_a_buble_sort);

const auto kAscendingGtestValues = ppc::util::TupleToGTestValues(kAscendingPerfTasks);

const auto kAscendingPerfTestName = NikitinABubleSortAscendingPerfTests::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunPerfAscending, NikitinABubleSortAscendingPerfTests, kAscendingGtestValues,
                         kAscendingPerfTestName);

// Тесты с данными по убыванию
TEST_P(NikitinABubleSortDescendingPerfTests, RunPerfModesDescending) {
  ExecuteTest(GetParam());
}

const auto kDescendingPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, NikitinABubleSortMPI, NikitinABubleSortSEQ>(PPC_SETTINGS_nikitin_a_buble_sort);

const auto kDescendingGtestValues = ppc::util::TupleToGTestValues(kDescendingPerfTasks);

const auto kDescendingPerfTestName = NikitinABubleSortDescendingPerfTests::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunPerfDescending, NikitinABubleSortDescendingPerfTests, kDescendingGtestValues,
                         kDescendingPerfTestName);

}  // namespace

}  // namespace nikitin_a_buble_sort
