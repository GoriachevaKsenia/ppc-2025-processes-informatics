#include <gtest/gtest.h>
#include <mpi.h>

#include <array>
#include <cstddef>
#include <string>
#include <tuple>
#include <vector>

#include "kondakov_v_reduce/common/include/common.hpp"
#include "kondakov_v_reduce/mpi/include/ops_mpi.hpp"
#include "kondakov_v_reduce/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace kondakov_v_reduce {

class KondakovVReduceRunFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &param) {
    return std::get<1>(param);
  }

 protected:
  void SetUp() override {
    const auto &params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());

    values_ = std::get<0>(params);
    op_name_ = std::get<1>(params);

    if (op_name_ == "sum") {
      op_ = ReduceOp::kSum;
    } else if (op_name_ == "prod") {
      op_ = ReduceOp::kProd;
    } else if (op_name_ == "min") {
      op_ = ReduceOp::kMin;
    } else if (op_name_ == "max") {
      op_ = ReduceOp::kMax;
    }
  }

  InType GetTestInputData() override {
    return InType{values_, op_};
  }

  bool CheckTestOutputData(OutType &output) override {
    int mpi_initialized = 0;
    MPI_Initialized(&mpi_initialized);

    if (mpi_initialized != 0) {
      int rank = 0;
      MPI_Comm_rank(MPI_COMM_WORLD, &rank);
      if (rank != 0) {
        return true;
      }
    }

    if (values_.empty()) {
      return output == GetNeutralElement(op_);
    }

    OutType expected = values_[0];
    for (std::size_t i = 1; i < values_.size(); ++i) {
      expected = ApplyReduceOp(expected, values_[i], op_);
    }
    return output == expected;
  }

 private:
  std::vector<int> values_;
  std::string op_name_;
  ReduceOp op_ = ReduceOp::kSum;
};

namespace {

TEST_P(KondakovVReduceRunFuncTests, ReduceCorrectness) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 4> kTestParams = {
    {{{1, 2, 3, 4}, "sum"}, {{2, 3, 4}, "prod"}, {{7, 2, 9, 1}, "min"}, {{4, 8, 6}, "max"}}};

const auto kTestTasksList =
    std::tuple_cat(ppc::util::AddFuncTask<KondakovVReduceTaskMPI, InType>(kTestParams, PPC_SETTINGS_kondakov_v_reduce),
                   ppc::util::AddFuncTask<KondakovVReduceTaskSEQ, InType>(kTestParams, PPC_SETTINGS_kondakov_v_reduce));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName = KondakovVReduceRunFuncTests::PrintFuncTestName<KondakovVReduceRunFuncTests>;

INSTANTIATE_TEST_SUITE_P(KondakovVReduceTests, KondakovVReduceRunFuncTests, kGtestValues, kPerfTestName);

}  // namespace
}  // namespace kondakov_v_reduce
