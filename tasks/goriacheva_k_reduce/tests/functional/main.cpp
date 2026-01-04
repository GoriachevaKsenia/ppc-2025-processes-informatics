#include <gtest/gtest.h>

#include <cstddef>
#include <fstream>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <vector>

#include "goriacheva_k_reduce/common/include/common.hpp"
#include "goriacheva_k_reduce/mpi/include/ops_mpi.hpp"
#include "goriacheva_k_reduce/seq/include/ops_seq.hpp"
#include "task/include/task.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace goriacheva_k_reduce {

using FuncParam = ppc::util::FuncTestParam<InType, OutType, TestType>;

class GoriachevaKReduceFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const testing::TestParamInfo<FuncParam> &info) {
    return std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kNameTest)>(info.param);
  }

 protected:
  void SetUp() override {
    const auto &params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    input_ = std::get<0>(params);
    expected_ = std::get<1>(params);
  }

  InType GetTestInputData() final {
    return input_;
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return output_data == expected_;
  }

 private:
  InType input_;
  OutType expected_;
};

namespace {

std::vector<FuncParam> LoadTestParams() {
  const std::string path = ppc::util::GetAbsoluteTaskPath(PPC_ID_goriacheva_k_reduce, "tests.json");

  std::ifstream fin(path);
  if (!fin.is_open()) {
    throw std::runtime_error("Cannot open tests.json");
  }

  nlohmann::json j;
  fin >> j;

  std::vector<FuncParam> cases;
  cases.reserve(j.size() * 2);

  const std::string settings_path = PPC_SETTINGS_goriacheva_k_reduce;
  const std::string mpi_suffix =
      ppc::task::GetStringTaskType(GoriachevaKReduceMPI::GetStaticTypeOfTask(), settings_path);
  const std::string seq_suffix =
      ppc::task::GetStringTaskType(GoriachevaKReduceSEQ::GetStaticTypeOfTask(), settings_path);

  for (const auto &item : j) {
    TestType tc{item.at("input").get<InType>(), item.at("result").get<OutType>(), item.at("name").get<std::string>()};

    std::string mpi_name = std::get<2>(tc) + "_" + mpi_suffix;
    cases.emplace_back(ppc::task::TaskGetter<GoriachevaKReduceMPI, InType>, mpi_name, tc);

    std::string seq_name = std::get<2>(tc) + "_" + seq_suffix;
    cases.emplace_back(ppc::task::TaskGetter<GoriachevaKReduceSEQ, InType>, seq_name, tc);
  }

  return cases;
}

const std::vector<FuncParam> kFuncParams = LoadTestParams();

TEST_P(GoriachevaKReduceFuncTests, ReduceSum) {
  ExecuteTest(GetParam());
}

INSTANTIATE_TEST_SUITE_P(GoriachevaKReduceFunctionalTests, GoriachevaKReduceFuncTests, testing::ValuesIn(kFuncParams),
                         GoriachevaKReduceFuncTests::PrintTestParam);

}  // namespace

}  // namespace goriacheva_k_reduce
