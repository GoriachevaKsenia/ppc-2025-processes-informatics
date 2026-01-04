#include <gtest/gtest.h>
#include <mpi.h>

#include <array>
#include <string>
#include <tuple>
#include <vector>

#include "shkrebko_m_hypercube/common/include/common.hpp"
#include "shkrebko_m_hypercube/mpi/include/ops_mpi.hpp"
#include "shkrebko_m_hypercube/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"

namespace shkrebko_m_hypercube {

class ShkrebkoMHypercubeFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return std::to_string(std::get<0>(test_param)) + "_" + std::get<1>(test_param);
  }

 protected:
  void SetUp() override {
    TestType test_params = std::get<2>(GetParam());
    int value = std::get<0>(test_params);

    int world_size = 1;
    int mpi_initialized = 0;
    MPI_Initialized(&mpi_initialized);
    if (mpi_initialized != 0) {
      MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    }

    int destination = (world_size > 1) ? 1 : 0;
    input_data_ = {value, destination};
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return output_data.value == input_data_[0] && !output_data.path.empty();
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_ = {0, 0};
};

TEST_P(ShkrebkoMHypercubeFuncTests, HypercubeRouting) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 3> kTestParam = {std::make_tuple(10, "basic"), std::make_tuple(42, "medium"),
                                            std::make_tuple(100, "large")};

const auto kTestTasksList = std::tuple_cat(
    ppc::util::AddFuncTask<ShkrebkoMHypercubeMPI, InType>(kTestParam, PPC_SETTINGS_shkrebko_m_hypercube),
    ppc::util::AddFuncTask<ShkrebkoMHypercubeSEQ, InType>(kTestParam, PPC_SETTINGS_shkrebko_m_hypercube));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName = ShkrebkoMHypercubeFuncTests::PrintFuncTestName<ShkrebkoMHypercubeFuncTests>;

INSTANTIATE_TEST_SUITE_P(ShkrebkoMHypercube, ShkrebkoMHypercubeFuncTests, kGtestValues, kPerfTestName);

}  // namespace shkrebko_m_hypercube
