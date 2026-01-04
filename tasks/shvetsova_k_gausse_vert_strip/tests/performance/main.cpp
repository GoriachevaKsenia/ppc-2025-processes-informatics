#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <vector>

#include "shvetsova_k_gausse_vert_strip/common/include/common.hpp"
#include "shvetsova_k_gausse_vert_strip/mpi/include/ops_mpi.hpp"
#include "shvetsova_k_gausse_vert_strip/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace shvetsova_k_gausse_vert_strip {

class ShvetsovaKGaussVertStripRunPerfTestProcesses : public ppc::util::BaseRunPerfTests<InType, OutType> {
 protected:
  void SetUp() override {
    // Размер матрицы для замера производительности (1000x1000)

    const int sz = 500;

    // Полуширина ленты

    const int size_of_rib = 5;
    std::vector<std::vector<double>> matrix(sz, std::vector<double>(sz, 0.0));
    std::vector<double> b(sz, 0.0);

    // Генерация ленточной матрицы с диагональным преобладанием
    // Это гарантирует, что решение существует и метод Гаусса будет устойчив

    for (int i = 0; i < sz; ++i) {
      double row_sum = 0.0;

      for (int j = std::max(0, i - size_of_rib + 1); j <= std::min(sz - 1, i + size_of_rib - 1); ++j) {
        if (i != j) {
          matrix[i][j] = ((i + j) % 5) + 1.0;  // Заполнение вне диагонали

          row_sum += std::abs(matrix[i][j]);
        }
      }

      matrix[i][i] = row_sum + 10.0;  // Диагональное преобладание

      // Задаем правую часть так, чтобы ответом был вектор из единиц (x[i] = 1.0)
      double b_val = 0.0;

      for (int j = 0; j < sz; ++j) {
        b_val += matrix[i][j];
      }

      b[i] = b_val;
    }

    input_data_ = {matrix, b};
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return !output_data.empty() && output_data.size() == input_data_.second.size();
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
};

// Регистрация задач

TEST_P(ShvetsovaKGaussVertStripRunPerfTestProcesses, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, ShvetsovaKGaussVertStripMPI, ShvetsovaKGaussVertStripSEQ>(
        PPC_SETTINGS_shvetsova_k_gausse_vert_strip);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = ShvetsovaKGaussVertStripRunPerfTestProcesses::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, ShvetsovaKGaussVertStripRunPerfTestProcesses, kGtestValues, kPerfTestName);

}  // namespace shvetsova_k_gausse_vert_strip
