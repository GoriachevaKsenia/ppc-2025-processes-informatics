#include <gtest/gtest.h>
#include <mpi.h>

#include <array>
#include <cstddef>
#include <numeric>
#include <random>
#include <string>
#include <tuple>
#include <vector>

#include "../../common/include/common.hpp"
#include "../../mpi/include/reduce_mpi.hpp"
#include "../../seq/include/reduce_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace kutergin_v_reduce {

TEST(KuterginReduceValidation, SeqHandlesEmptyVector) {
  InType empty_input{{}, 0};
  ReduceSequential task_empty(empty_input);
  ASSERT_TRUE(task_empty.Validation());
  ASSERT_TRUE(task_empty.Run());
  ASSERT_EQ(task_empty.GetOutput(), 0);
}

class ReduceFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param)  // вывод имен тестов
  {
    return "vec_size_" + std::to_string(std::get<0>(test_param)) + "_root_" + std::to_string(std::get<1>(test_param));
  }

 protected:
  // подготовка входных данных перед каждым тестом
  void SetUp() override {
    int rank = 0;
    if (ppc::util::IsUnderMpirun()) {
      MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    }

    // получение параметров теста
    const auto &params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    int vec_size = std::get<0>(params);
    int root_proc = std::get<1>(params);

    // генерация входных данных
    std::mt19937 gen(rank);  // rank выступает как seed для воспроизводимости
    std::uniform_int_distribution<> dis(1, 10);
    std::vector<int> local_vec(vec_size);
    for (int i = 0; i < vec_size; ++i) {
      local_vec[i] = dis(gen);
    }

    input_data_ = InType{local_vec, root_proc};
  }

  // проверка результата после выполнения задачи
  bool CheckTestOutputData(OutType &output_data) final {
    const auto &test_name = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kNameTest)>(GetParam());
    const auto &input = GetTestInputData();  // получение структуры {data, root}

    if (test_name.find("_seq_") != std::string::npos) {
      const auto &input_vec = input.data;
      OutType expected_sum = std::accumulate(input_vec.begin(), input_vec.end(), 0);

      /*
      std::cout << "-> SEQ ACTUAL:   " << output_data << std::endl;
      std::cout << "-> SEQ EXPECTED: " << expected_sum << std::endl;
      */

      return output_data == expected_sum;
    }

    if (test_name.find("_mpi_") != std::string::npos) {
      int rank = 0;
      MPI_Comm_rank(MPI_COMM_WORLD, &rank);

      // корневой процесс возвращает результат проверки
      if (rank == input.root) {
        // построение буфера для сбора
        int world_size = 0;
        MPI_Comm_size(MPI_COMM_WORLD, &world_size);

        int expected_sum = 0;
        for (int rank_i = 0; rank_i < world_size; ++rank_i) {
          std::mt19937 gen(rank_i);
          std::uniform_int_distribution<> dis(1, 10);
          for (size_t i = 0; i < input.data.size(); ++i) {
            expected_sum += dis(gen);
          }
        }

        /*
        std::cout << "-> MPI ROOT ACTUAL:   " << output_data << std::endl;
        std::cout << "-> MPI ROOT EXPECTED: " << expected_sum << std::endl;
        */

        return output_data == expected_sum;
      }

      // рабочие процессы возвращают true
      return true;
    }
    return false;
  }

  // возврат подготовленных входных данных для задачи
  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_{};
};

namespace {

TEST_P(ReduceFuncTests, ReduceTest) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 4> kTestCases = {
    std::make_tuple(10, 0, "size_10_root_0"), std::make_tuple(20, 1, "size_20_root_1"),
    std::make_tuple(1, 3, "size_1_root_3"), std::make_tuple(100, 2, "size_100_root_2")};

const auto kTestTasksList =
    std::tuple_cat(ppc::util::AddFuncTask<ReduceSequential, InType>(kTestCases, PPC_SETTINGS_kutergin_v_reduce),
                   ppc::util::AddFuncTask<ReduceMPI, InType>(kTestCases, PPC_SETTINGS_kutergin_v_reduce));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kTestName = ReduceFuncTests::PrintFuncTestName<ReduceFuncTests>;

// NOLINTNEXTLINE(modernize-type-traits, cppcoreguidelines-avoid-non-const-global-variables)
INSTANTIATE_TEST_SUITE_P(Reduce, ReduceFuncTests, kGtestValues, kTestName);
}  // namespace

}  // namespace kutergin_v_reduce
