#include <gtest/gtest.h>
#include <mpi.h>

#include <array>
#include <cmath>
#include <string>
#include <tuple>
#include <vector>

#include "dolov_v_torus_topology/common/include/common.hpp"
#include "dolov_v_torus_topology/mpi/include/ops_mpi.hpp"
#include "dolov_v_torus_topology/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"

namespace dolov_v_torus_topology {

class DolovVTorusTopologyFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return std::to_string(std::get<0>(test_param)) + "_" + std::get<1>(test_param);
  }

 protected:
  void SetUp() override {
    int world_size = 1;
    int is_init = 0;
    MPI_Initialized(&is_init);
    if (is_init != 0) {
      MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    }

    TestType params = std::get<2>(GetParam());
    int test_type = std::get<0>(params);

    std::vector<int> message = {42, 13, 7};
    input_data_.sender_rank = 0;
    input_data_.total_procs = world_size;
    input_data_.message = message;

    int r = static_cast<int>(std::sqrt(world_size));
    while (world_size % r != 0) {
      r--;
    }
    int c = world_size / r;

    expected_.received_message = message;

    switch (test_type) {
      case 0:
        input_data_.receiver_rank = 0;
        expected_.route = {0};
        break;
      case 1:
        input_data_.receiver_rank = (c > 1) ? 1 : 0;
        expected_.route.clear();
        break;
      case 2:
        input_data_.receiver_rank = (r > 1) ? c : 0;
        expected_.route.clear();
        break;
      case 3:
        input_data_.receiver_rank = (c > 1) ? (c - 1) : 0;
        expected_.route.clear();
        break;
      case 4:
        input_data_.receiver_rank = (r > 1) ? (r - 1) * c : 0;
        expected_.route.clear();
        break;
      case 5:
        input_data_.receiver_rank = world_size - 1;
        expected_.route.clear();
        break;
      case 6:
        input_data_.sender_rank = world_size - 1;
        input_data_.receiver_rank = 0;
        expected_.route.clear();
        break;
      default:
        input_data_.receiver_rank = 0;
        expected_.route = {0};
        break;
    }
  }

  bool CheckTestOutputData(OutType &output_data) final {
    if (output_data.received_message != expected_.received_message) {
      return false;
    }
    if (output_data.route.empty()) {
      return false;
    }
    return output_data.route.front() == input_data_.sender_rank &&
           output_data.route.back() == input_data_.receiver_rank;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
  OutType expected_;
};

namespace {

TEST_P(DolovVTorusTopologyFuncTests, TorusRouting) {
  ExecuteTest(GetParam());
}
const std::array<TestType, 7> kTestParam = {std::make_tuple(0, "Self"),       std::make_tuple(1, "EastStep"),
                                            std::make_tuple(2, "SouthStep"),  std::make_tuple(3, "WestWrap"),
                                            std::make_tuple(4, "NorthWrap"),  std::make_tuple(5, "FarDestination"),
                                            std::make_tuple(6, "ReversePath")};

const auto kTestTasksList = std::tuple_cat(
    ppc::util::AddFuncTask<DolovVTorusTopologyMPI, InType>(kTestParam, PPC_SETTINGS_dolov_v_torus_topology),
    ppc::util::AddFuncTask<DolovVTorusTopologySEQ, InType>(kTestParam, PPC_SETTINGS_dolov_v_torus_topology));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

INSTANTIATE_TEST_SUITE_P(TorusFunctional, DolovVTorusTopologyFuncTests, kGtestValues,
                         DolovVTorusTopologyFuncTests::PrintFuncTestName<DolovVTorusTopologyFuncTests>);
}  // namespace
}  // namespace dolov_v_torus_topology
