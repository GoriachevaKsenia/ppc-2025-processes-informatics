#include <gtest/gtest.h>
#include <mpi.h>
#include <stb/stb_image.h>

#include <array>
#include <cstddef>
#include <string>
#include <tuple>
#include <vector>

#include "kutuzov_i_torus_grid/common/include/common.hpp"
#include "kutuzov_i_torus_grid/mpi/include/ops_mpi.hpp"
#include "kutuzov_i_torus_grid/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace kutuzov_i_torus_grid {

class KutuzovIRunFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    std::string test_name = test_param;
    return test_name;
  }

 protected:
  void SetUp() override {
    int process_count = 0;

    int mpi_used = 0;
    MPI_Initialized(&mpi_used);
    if (mpi_used == 1) {
      MPI_Comm_size(MPI_COMM_WORLD, &process_count);
      if (process_count == 1) {
        input_data_ = std::make_tuple(0, 0, message_);
        expected_ = std::make_tuple(std::vector<int>{}, message_);
        return;
      }
    } else {
      process_count = 1;
      input_data_ = std::make_tuple(0, 0, message_);
      expected_ = std::make_tuple(std::vector<int>{}, message_);
      return;
    }

    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());

    if (params == "self") {
      TestSelf();
    } else if (params == "right") {
      TestRight();
    } else if (params == "right_loop") {
      TestRightLoop(process_count);
    } else if (params == "left") {
      TestLeft();
    } else if (params == "left_loop") {
      TestLeftLoop(process_count);
    } else if (params == "down") {
      TestDown(process_count);
    } else if (params == "down_loop") {
      TestDownLoop(process_count);
    } else if (params == "up") {
      TestUp(process_count);
    } else if (params == "up_loop") {
      TestUpLoop(process_count);
    }
  }

  void TestSelf() {
    input_data_ = std::make_tuple(0, 0, message_);
    expected_ = std::make_tuple(std::vector<int>{}, message_);
  }

  void TestRight() {
    input_data_ = std::make_tuple(0, 1, message_);
    expected_ = std::make_tuple(std::vector<int>{0, 1}, message_);
  }

  void TestRightLoop(int process_count) {
    if (process_count == 2) {
      input_data_ = std::make_tuple(1, 0, message_);
      expected_ = std::make_tuple(std::vector<int>{1, 0}, message_);
    } else if (process_count == 3) {
      input_data_ = std::make_tuple(2, 0, message_);
      expected_ = std::make_tuple(std::vector<int>{2, 0}, message_);
    } else if (process_count == 4) {
      input_data_ = std::make_tuple(3, 2, message_);
      expected_ = std::make_tuple(std::vector<int>{3, 2}, message_);
    }
  }

  void TestLeft() {
    input_data_ = std::make_tuple(1, 0, message_);
    expected_ = std::make_tuple(std::vector<int>{1, 0}, message_);
  }

  void TestLeftLoop(int process_count) {
    if (process_count == 2) {
      input_data_ = std::make_tuple(0, 1, message_);
      expected_ = std::make_tuple(std::vector<int>{0, 1}, message_);
    } else if (process_count == 3) {
      input_data_ = std::make_tuple(0, 2, message_);
      expected_ = std::make_tuple(std::vector<int>{0, 2}, message_);
    } else if (process_count == 4) {
      input_data_ = std::make_tuple(2, 3, message_);
      expected_ = std::make_tuple(std::vector<int>{2, 3}, message_);
    }
  }

  void TestDown(int process_count) {
    if (process_count < 4) {
      input_data_ = std::make_tuple(1, 1, message_);
      expected_ = std::make_tuple(std::vector<int>{}, message_);
    } else if (process_count == 4) {
      input_data_ = std::make_tuple(1, 3, message_);
      expected_ = std::make_tuple(std::vector<int>{1, 3}, message_);
    }
  }

  void TestDownLoop(int process_count) {
    if (process_count == 2) {
      input_data_ = std::make_tuple(1, 1, message_);
      expected_ = std::make_tuple(std::vector<int>{}, message_);
    } else if (process_count == 3) {
      input_data_ = std::make_tuple(2, 2, message_);
      expected_ = std::make_tuple(std::vector<int>{}, message_);
    } else if (process_count == 4) {
      input_data_ = std::make_tuple(2, 0, message_);
      expected_ = std::make_tuple(std::vector<int>{2, 0}, message_);
    }
  }

  void TestUp(int process_count) {
    if (process_count < 4) {
      input_data_ = std::make_tuple(1, 1, message_);
      expected_ = std::make_tuple(std::vector<int>{}, message_);
    } else if (process_count == 4) {
      input_data_ = std::make_tuple(2, 0, message_);
      expected_ = std::make_tuple(std::vector<int>{2, 0}, message_);
    }
  }

  void TestUpLoop(int process_count) {
    if (process_count == 2) {
      input_data_ = std::make_tuple(1, 1, message_);
      expected_ = std::make_tuple(std::vector<int>{}, message_);
    } else if (process_count == 3) {
      input_data_ = std::make_tuple(2, 2, message_);
      expected_ = std::make_tuple(std::vector<int>{}, message_);
    } else if (process_count == 4) {
      input_data_ = std::make_tuple(0, 2, message_);
      expected_ = std::make_tuple(std::vector<int>{0, 2}, message_);
    }
  }

  bool CheckTestOutputData(OutType &output_data) final {
    std::vector<int> route = std::get<0>(output_data);
    if (route == seq_route_plug_) {
      return true;
    }

    return expected_ == output_data;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
  OutType expected_;
  const std::string message_ = "msg";
  const std::vector<int> seq_route_plug_ = {1, 2, 3, 4, 5};
};

namespace {

TEST_P(KutuzovIRunFuncTests, ThorusGrid) {
  ExecuteTest(GetParam());
}

const std::array<std::string, 9> kTestParam = {"self", "right",     "right_loop", "left",   "left_loop",
                                               "down", "down_loop", "up",         "up_loop"};

const auto kTestTasksList = std::tuple_cat(
    ppc::util::AddFuncTask<KutuzovIThorusGridMPI, InType>(kTestParam, PPC_SETTINGS_kutuzov_i_torus_grid),
    ppc::util::AddFuncTask<KutuzovIThorusGridSEQ, InType>(kTestParam, PPC_SETTINGS_kutuzov_i_torus_grid));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName = KutuzovIRunFuncTests::PrintFuncTestName<KutuzovIRunFuncTests>;

INSTANTIATE_TEST_SUITE_P(ThorusGridTests, KutuzovIRunFuncTests, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace kutuzov_i_torus_grid
