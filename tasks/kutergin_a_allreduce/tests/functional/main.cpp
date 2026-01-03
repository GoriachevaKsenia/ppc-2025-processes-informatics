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
#include "../../mpi/include/ops_mpi.hpp"
#include "../../seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace kutergin_a_allreduce {

TEST(KuterginAllreduceValidation, SeqHandlesEmptyVector) {
  InType empty_input{{}, 0};
  AllreduceSequential task_empty(empty_input);
  ASSERT_TRUE(task_empty.Validation());
  ASSERT_TRUE(task_empty.PreProcessing());
  ASSERT_TRUE(task_empty.Run());
  ASSERT_EQ(task_empty.GetOutput(), 0);
}

class AllreduceFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return "vec_size_" + std::to_string(std::get<0>(test_param)) + "_root_" + std::to_string(std::get<1>(test_param));
  }

 protected:
  void SetUp() override {
    int rank = 0;
    if (ppc::util::IsUnderMpirun()) {
      MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    }

    const auto &params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    int vec_size = std::get<0>(params);
    int root_proc = std::get<1>(params);

    std::mt19937 gen(rank);
    std::uniform_int_distribution<> dis(1, 10);
    std::vector<int> local_vec(vec_size);
    for (int i = 0; i < vec_size; ++i) {
      local_vec[i] = dis(gen);
    }

    input_data_ = InType{local_vec, root_proc};
  }

  bool CheckTestOutputData(OutType &output_data) final {
    const auto &test_name = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kNameTest)>(GetParam());
    const auto &input = GetTestInputData();

    if (test_name.find("_seq_") != std::string::npos) {
      OutType expected_sum = std::accumulate(input.elements.begin(), input.elements.end(), 0);
      return output_data == expected_sum;
    }

    if (test_name.find("_mpi_") != std::string::npos) {
      int world_size = 0;
      MPI_Comm_size(MPI_COMM_WORLD, &world_size);

      int expected_global_sum = 0;
      for (int rank_idx = 0; rank_idx < world_size; ++rank_idx) {
        std::mt19937 gen_r(rank_idx);
        std::uniform_int_distribution<> dis_r(1, 10);
        for (size_t i = 0; i < input.elements.size(); ++i) {
          expected_global_sum += dis_r(gen_r);
        }
      }

      return output_data == expected_global_sum;
    }
    return false;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_{};
};

namespace {

TEST_P(AllreduceFuncTests, AllreduceTest) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 4> kTestCases = {
    std::make_tuple(10, 0, "size_10_root_0"), std::make_tuple(20, 1, "size_20_root_1"),
    std::make_tuple(1, 3, "size_1_root_3"), std::make_tuple(100, 2, "size_100_root_2")};

const auto kTestTasksList =
    std::tuple_cat(ppc::util::AddFuncTask<AllreduceSequential, InType>(kTestCases, PPC_SETTINGS_kutergin_a_allreduce),
                   ppc::util::AddFuncTask<AllreduceMPI, InType>(kTestCases, PPC_SETTINGS_kutergin_a_allreduce));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kTestName = AllreduceFuncTests::PrintFuncTestName<AllreduceFuncTests>;

INSTANTIATE_TEST_SUITE_P(Allreduce, AllreduceFuncTests, kGtestValues, kTestName);

}  // namespace
}  // namespace kutergin_a_allreduce
