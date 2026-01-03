#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

#include "leonova_a_star/common/include/common.hpp"
#include "leonova_a_star/mpi/include/ops_mpi.hpp"
#include "leonova_a_star/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace leonova_a_star {

class LeonovaAStarRunFuncTestsProcesses : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    static int counter = 0;
    const auto &input_pair = std::get<0>(test_param);
    const auto &matrix_a = std::get<0>(input_pair);
    const auto &matrix_b = std::get<1>(input_pair);

    std::stringstream ss;
    ss << "Test_" << ++counter << "_A_" << matrix_a.size() << "x" << matrix_a[0].size() << "_B_" << matrix_b.size()
       << "x" << matrix_b[0].size();
    return ss.str();
  }

 protected:
  void SetUp() override {
    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());

    input_data_ = std::get<0>(params);
    expected_output_ = std::get<1>(params);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    if (output_data.size() != expected_output_.size()) {
      return false;
    }

    for (size_t index = 0; index < output_data.size(); index++) {
      if (output_data[index].size() != expected_output_[index].size()) {
        return false;
      }

      for (size_t jndex = 0; jndex < output_data[index].size(); jndex++) {
        if (output_data[index][jndex] != expected_output_[index][jndex]) {
          return false;
        }
      }
    }

    return true;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;        // Входные данные: пара матриц (A, B)
  OutType expected_output_;  // Ожидаемый результат: матрица C
};

namespace {

TEST_P(LeonovaAStarRunFuncTestsProcesses, RunFuncTests) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 6> kTestParam = {
    // Тест 1: Единичная матрица 2x2
    std::make_tuple(std::make_tuple(std::vector<std::vector<int>>{{1, 0}, {0, 1}},  // A
                                    std::vector<std::vector<int>>{{5, 6}, {7, 8}}   // B
                                    ),
                    std::vector<std::vector<int>>{{5, 6}, {7, 8}}  // Ожидаемый результат
                    ),

    // Тест 2: Нулевая матрица 2x2
    std::make_tuple(std::make_tuple(std::vector<std::vector<int>>{{0, 0}, {0, 0}},  // A
                                    std::vector<std::vector<int>>{{1, 2}, {3, 4}}   // B
                                    ),
                    std::vector<std::vector<int>>{{0, 0}, {0, 0}}  // Ожидаемый результат
                    ),

    // Тест 3: Матрицы 3x3
    std::make_tuple(std::make_tuple(std::vector<std::vector<int>>{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}},  // A
                                    std::vector<std::vector<int>>{{9, 8, 7}, {6, 5, 4}, {3, 2, 1}}   // B
                                    ),
                    std::vector<std::vector<int>>{{30, 24, 18}, {84, 69, 54}, {138, 114, 90}}  // Ожидаемый результат
                    ),

    // Тест 4: Неквадратные матрицы 2x3 и 3x2
    std::make_tuple(std::make_tuple(std::vector<std::vector<int>>{{1, 0, -1}, {2, 1, 0}},  // A (2x3)
                                    std::vector<std::vector<int>>{{2, 1}, {1, 2}, {0, 3}}  // B (3x2)
                                    ),
                    std::vector<std::vector<int>>{{2, -2}, {5, 4}}  // Ожидаемый результат (2x2)
                    ),

    // Тест 5: Матрица-строка и матрица-столбец
    std::make_tuple(std::make_tuple(std::vector<std::vector<int>>{{1, 2, 3}},     // A (1x3)
                                    std::vector<std::vector<int>>{{4}, {5}, {6}}  // B (3x1)
                                    ),
                    std::vector<std::vector<int>>{{32}}  // Ожидаемый результат (1x1)
                    ),

    // Тест 6: Большие значения
    std::make_tuple(std::make_tuple(std::vector<std::vector<int>>{{100, 200}, {300, 400}},  // A
                                    std::vector<std::vector<int>>{{5, 6}, {7, 8}}           // B
                                    ),
                    std::vector<std::vector<int>>{{1900, 2200}, {4300, 5000}})};

const auto kTestTasksList =
    std::tuple_cat(ppc::util::AddFuncTask<LeonovaAStarMPI, InType>(kTestParam, PPC_SETTINGS_example_processes_2),
                   ppc::util::AddFuncTask<LeonovaAStarSEQ, InType>(kTestParam, PPC_SETTINGS_example_processes_2));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName = LeonovaAStarRunFuncTestsProcesses::PrintFuncTestName<LeonovaAStarRunFuncTestsProcesses>;

INSTANTIATE_TEST_SUITE_P(RunMatrixFunc, LeonovaAStarRunFuncTestsProcesses, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace leonova_a_star
