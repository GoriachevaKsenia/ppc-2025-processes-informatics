#include <gtest/gtest.h>
#include <mpi.h>

#include <array>
#include <cstddef>
#include <string>
#include <tuple>
#include <vector>

#include "ashihmin_d_scatter_trans_from_one_to_all/common/include/common.hpp"
#include "ashihmin_d_scatter_trans_from_one_to_all/mpi/include/ops_mpi.hpp"
#include "ashihmin_d_scatter_trans_from_one_to_all/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"

namespace ashihmin_d_scatter_trans_from_one_to_all {

class AshihminDRunFuncTestsProcesses : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    int count = std::get<0>(test_param);
    int root = std::get<1>(test_param);
    std::string type_str = std::get<2>(test_param);
    return type_str + "_Count" + std::to_string(count) + "_Root" + std::to_string(root);
  }

 protected:
  void SetUp() override {
    auto param_tuple = GetParam();
    auto test_params = std::get<2>(param_tuple);

    int count = std::get<0>(test_params);
    int root = std::get<1>(test_params);
    std::string type_str = std::get<2>(test_params);

    MPI_Datatype mpi_type = MPI_INT;
    if (type_str == "float") {
      mpi_type = MPI_FLOAT;
    } else if (type_str == "double") {
      mpi_type = MPI_DOUBLE;
    }

    int world_size = 4;

    ScatterParams params;
    params.elements_per_process = count;
    params.root = root;
    params.datatype = mpi_type;

    int total_elements = count * world_size;
    params.data.resize(total_elements);

    for (int i = 0; i < total_elements; ++i) {
      params.data[i] = i;
    }

    input_data_ = params;

    test_count_ = count;
    test_root_ = root;
    test_type_ = type_str;
  }

  bool CheckTestOutputData(OutType &output_data) final {
    auto param_tuple = GetParam();
    std::string description = std::get<1>(param_tuple);
    bool is_mpi = (description.find("mpi") != std::string::npos);

    if (!is_mpi) {
      if (test_root_ != 0) {
        return !output_data.empty();
      }

      if (output_data.size() != static_cast<size_t>(test_count_)) {
        return false;
      }

      for (int i = 0; i < test_count_; ++i) {
        if (output_data[i] != i) {
          return false;
        }
      }
      return true;
    }

    return output_data.size() == static_cast<size_t>(test_count_);
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_{};
  int test_count_ = 0;
  int test_root_ = -1;
  std::string test_type_;
};

namespace {

TEST_P(AshihminDRunFuncTestsProcesses, ScatterTest) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 9> kTestParam = {
    std::make_tuple(100, 0, "int"),   std::make_tuple(100, 1, "int"),     std::make_tuple(100, 2, "int"),
    std::make_tuple(1000, 0, "int"),  std::make_tuple(1000, 1, "float"),  std::make_tuple(1000, 2, "double"),
    std::make_tuple(10000, 0, "int"), std::make_tuple(10000, 1, "float"), std::make_tuple(10000, 2, "double")};

const auto kTestTasksList = std::tuple_cat(ppc::util::AddFuncTask<AshihminDScatterTransFromOneToAllMPI<int>, InType>(
                                               kTestParam, PPC_SETTINGS_ashihmin_d_scatter_trans_from_one_to_all),
                                           ppc::util::AddFuncTask<AshihminDScatterTransFromOneToAllSEQ<int>, InType>(
                                               kTestParam, PPC_SETTINGS_ashihmin_d_scatter_trans_from_one_to_all));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName = AshihminDRunFuncTestsProcesses::PrintFuncTestName<AshihminDRunFuncTestsProcesses>;

INSTANTIATE_TEST_SUITE_P(ScatterTests, AshihminDRunFuncTestsProcesses, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace ashihmin_d_scatter_trans_from_one_to_all
