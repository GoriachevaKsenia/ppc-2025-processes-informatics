#include <gtest/gtest.h>
#include <mpi.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <string>
#include <tuple>
#include <vector>

#include "../../common/include/common.hpp"
#include "../../mpi/include/linear_contrast_stretching_mpi.hpp"
#include "../../seq/include/linear_contrast_stretching_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace kutergin_v_linear_contrast_stretching {

class LinearContrastStretchingFuncTests
    : public ppc::util::BaseRunFuncTests<InType, OutType,
                                         TestType>  // наследник ppc::util::BaseRunFuncTests<InType, OutType, TestType>
{
 public:
  static std::string PrintTestParam(const TestType &test_param)  // вывод имен тестов
  {
    return std::get<2>(test_param);
  }

 protected:
  // подготовка входных данных перед каждым тестом
  void SetUp() override {
    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(
        GetParam());                         // получение параметров для текущего теста
    input_data_ = std::get<0>(params);       // извлечение входных данных
    expected_output_ = std::get<1>(params);  // извлечение ожидаемого результата
  }

  // проверка результата после выполнения задачи
  bool CheckTestOutputData(OutType &output_data) final {
    int process_rank = 0;
    if (ppc::util::IsUnderMpirun()) {
      MPI_Comm_rank(MPI_COMM_WORLD, &process_rank);
    }
    if (process_rank == 0)  // проверку выполняет только нулевой процесс
    {
      std::cout << "-> RANK 0 EXPECTED: ";
      PrintVector(expected_output_);
      std::cout << '\n';
      std::cout << "-> RANK 0 ACTUAL:   ";
      PrintVector(output_data);
      std::cout << '\n';
      return output_data == expected_output_;
    }
    return true;  // у всех остальных процессов все в порядке
  }

  // возврат подготовленных входных данных для задачи
  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
  OutType expected_output_;

  static void PrintVector(const std::vector<unsigned char> &vec) {
    std::cout << "{";
    for (size_t i = 0; i < vec.size(); ++i) {
      std::cout << static_cast<int>(vec[i]) << (i == vec.size() - 1 ? "" : ", ");
    }
    std::cout << "}";
  }
};

namespace  // анонимное пространство имен
{

TEST_P(LinearContrastStretchingFuncTests, LinearContrastStretchingTest)  // параметризованный тест
{
  ExecuteTest(GetParam());
}

// массив с наборами тестовых данных
const std::array<TestType, 3> kTestCases = {
    // Успешные тесты
    std::make_tuple(InType{{100, 110, 125, 140, 150}, 5, 1}, OutType{0, 51, 127, 204, 255}, "basic_stretch"),
    std::make_tuple(InType{{0, 50, 128, 204, 255}, 5, 1}, OutType{0, 50, 128, 204, 255}, "full_range"),
    std::make_tuple(InType{{80, 80, 80, 80, 80}, 5, 1}, OutType{80, 80, 80, 80, 80}, "single_color")};

// используем фреймворк для подготовки задач к запуску
const auto kTestTasksList = std::tuple_cat(ppc::util::AddFuncTask<LinearContrastStretchingSequential, InType>(
                                               kTestCases, PPC_SETTINGS_kutergin_v_linear_contrast_stretching),
                                           ppc::util::AddFuncTask<LinearContrastStretchingMPI, InType>(
                                               kTestCases, PPC_SETTINGS_kutergin_v_linear_contrast_stretching));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kTestName = LinearContrastStretchingFuncTests::PrintFuncTestName<LinearContrastStretchingFuncTests>;

// "регистрация" набора тестов и параметров в GTest
// NOLINTNEXTLINE(modernize-type-traits, cppcoreguidelines-avoid-non-const-global-variables)
INSTANTIATE_TEST_SUITE_P(LinearContrastStretching, LinearContrastStretchingFuncTests, kGtestValues, kTestName);
}  // namespace

}  // namespace kutergin_v_linear_contrast_stretching
