#include <gtest/gtest.h>

#include <cstddef>
#include <fstream>
#include <string>

#include "rozenberg_a_radix_simple_merge/common/include/common.hpp"
#include "rozenberg_a_radix_simple_merge/mpi/include/ops_mpi.hpp"
#include "rozenberg_a_radix_simple_merge/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"
#include "util/include/util.hpp"

namespace rozenberg_a_radix_simple_merge {

class RozenbergARadixSimpleMergePerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
  void SetUp() override {
    input_data_.clear();
    output_data_.clear();
    if (CheckTestAndRank()) {
      std::string abs_path = ppc::util::GetAbsoluteTaskPath(PPC_ID_rozenberg_a_radix_simple_merge, "perf_test.txt");
      std::ifstream file(abs_path);

      if (file.is_open()) {
        int size = 0;
        file >> size;

        InType input_data(size);
        for (int i = 0; i < size; i++) {
          file >> input_data[i];
        }

        OutType output_data(size);
        for (int i = 0; i < size; i++) {
          file >> output_data[i];
        }
        input_data_ = input_data;
        output_data_ = output_data;
      }
    }
  }

  bool CheckTestOutputData(OutType &output_data) final {
    if (CheckTestAndRank()) {
      return (output_data_ == output_data);
    }
    return true;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
  OutType output_data_;

  static bool CheckTestAndRank() {
    const std::string &test_name =
        std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kNameTest)>(GetParam());
    return (test_name.find("mpi") == std::string::npos || !ppc::util::IsUnderMpirun() || ppc::util::GetMPIRank() == 0);
  }
};

TEST_P(RozenbergARadixSimpleMergePerfTests, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, RozenbergARadixSimpleMergeMPI, RozenbergARadixSimpleMergeSEQ>(
        PPC_SETTINGS_rozenberg_a_radix_simple_merge);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = RozenbergARadixSimpleMergePerfTests::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, RozenbergARadixSimpleMergePerfTests, kGtestValues, kPerfTestName);

}  // namespace rozenberg_a_radix_simple_merge
