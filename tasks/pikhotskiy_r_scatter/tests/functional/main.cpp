#include <gtest/gtest.h>
#include <mpi.h>

#include <array>
#include <cstddef>
#include <cstring>
#include <string>
#include <tuple>
#include <vector>

#include "pikhotskiy_r_scatter/common/include/common.hpp"
#include "pikhotskiy_r_scatter/mpi/include/ops_mpi.hpp"
#include "pikhotskiy_r_scatter/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace pikhotskiy_r_scatter {

class PikhotskiyRScatterRunFuncTestsProcesses : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  PikhotskiyRScatterRunFuncTestsProcesses() = default;

  static std::string PrintTestParam(const TestType &test_param) {
    return std::get<1>(test_param);
  }

  static bool IsSeqTest() {
    const auto *test_info = ::testing::UnitTest::GetInstance()->current_test_info();
    if (test_info == nullptr) {
      return false;
    }
    std::string test_name = test_info->name();
    return test_name.find("seq") != std::string::npos || test_name.find("SEQ") != std::string::npos;
  }

 protected:
  void SetUp() override {
    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());

    input_data_root_ = std::get<0>(params);

    const void *sendbuf = std::get<0>(input_data_root_);
    int sendcount = std::get<1>(input_data_root_);
    MPI_Datatype sendtype = std::get<2>(input_data_root_);
    int recvcount = std::get<4>(input_data_root_);
    MPI_Datatype recvtype = std::get<5>(input_data_root_);
    int root = std::get<6>(input_data_root_);
    MPI_Comm comm = std::get<7>(input_data_root_);

    int rank = 0;
    int size = 0;

    size_t type_size = 0;

    size_t block_size = 0;

    const void *current_sendbuf = nullptr;
    if (IsSeqTest()) {
      if (recvtype == MPI_INT) {
        type_size = sizeof(int);
      } else if (recvtype == MPI_FLOAT) {
        type_size = sizeof(float);
      } else if (recvtype == MPI_DOUBLE) {
        type_size = sizeof(double);
      }
      current_sendbuf = sendbuf;
    } else {
      MPI_Comm_rank(comm, &rank);
      MPI_Comm_size(comm, &size);
      int mpi_type_size = 0;
      MPI_Type_size(recvtype, &mpi_type_size);
      type_size = static_cast<size_t>(mpi_type_size);
      current_sendbuf = (rank == root) ? sendbuf : nullptr;
    }

    block_size = static_cast<size_t>(recvcount) * type_size;

    recvbuf_storage_.resize(block_size);

    expected_storage_.resize(block_size);

    if (IsSeqTest()) {
      if (block_size > 0) {
        if (sendbuf != nullptr) {
          const auto *source = static_cast<const unsigned char *>(sendbuf);
          std::memcpy(expected_storage_.data(), source, block_size);
        } else {
          std::memset(expected_storage_.data(), 0, block_size);
        }
      }
    } else {
      if (block_size > 0) {
        std::vector<unsigned char> temp_buffer(block_size);
        MPI_Scatter(sendbuf, recvcount, recvtype, temp_buffer.data(), recvcount, recvtype, root, comm);
        std::memcpy(expected_storage_.data(), temp_buffer.data(), block_size);
      }
    }

    input_data_ = std::make_tuple(current_sendbuf,          // sendbuf
                                  sendcount,                // sendcount
                                  sendtype,                 // sendtype
                                  recvbuf_storage_.data(),  // recvbuf
                                  recvcount,                // recvcount
                                  recvtype,                 // recvtype
                                  root,                     // root
                                  comm                      // comm
    );

    expected_data_ = expected_storage_.data();
  }

  bool CheckTestOutputData(OutType &output_data) final {
    int recvcount = std::get<4>(input_data_);
    MPI_Datatype recvtype = std::get<5>(input_data_);

    size_t type_size = 0;
    if (IsSeqTest()) {
      if (recvtype == MPI_INT) {
        type_size = sizeof(int);
      } else if (recvtype == MPI_FLOAT) {
        type_size = sizeof(float);
      } else if (recvtype == MPI_DOUBLE) {
        type_size = sizeof(double);
      }
    } else {
      int mpi_type_size = 0;
      MPI_Type_size(recvtype, &mpi_type_size);
      type_size = static_cast<size_t>(mpi_type_size);
    }

    size_t total_bytes = static_cast<size_t>(recvcount) * type_size;

    if (total_bytes == 0) {
      return true;
    }

    return std::memcmp(output_data, expected_data_, total_bytes) == 0;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_root_;
  InType input_data_;
  OutType expected_data_ = nullptr;

  std::vector<unsigned char> recvbuf_storage_;
  std::vector<unsigned char> expected_storage_;
};

namespace {

const std::vector<int> kIntData = {1, 2, 3, 4, 5, 6};
const std::vector<float> kFloatData = {1.1F, 2.2F, 3.3F, 4.4F, 5.5F, 6.6F};
const std::vector<double> kDoubleData = {1.11, 2.22, 3.33, 4.44, 5.55, 6.66};

std::array<TestType, 4> CreateTestParams() {
  static std::vector<int> int_recv_buf(2);
  static std::vector<float> float_recv_buf(2);
  static std::vector<double> double_recv_buf(2);
  static std::vector<int> empty_recv_buf(0);

  static std::vector<int> int_expected_buf(2);
  static std::vector<float> float_expected_buf(2);
  static std::vector<double> double_expected_buf(2);
  static std::vector<int> empty_expected_buf(0);

  return {std::make_tuple(std::make_tuple(kIntData.data(),      // sendbuf
                                          2,                    // sendcount
                                          MPI_INT,              // sendtype
                                          int_recv_buf.data(),  // recvbuf
                                          2,                    // recvcount
                                          MPI_INT,              // recvtype
                                          0,                    // root
                                          MPI_COMM_WORLD        // comm
                                          ),
                          "test_int", int_expected_buf.data()),

          std::make_tuple(std::make_tuple(kFloatData.data(),      // sendbuf
                                          2,                      // sendcount
                                          MPI_FLOAT,              // sendtype
                                          float_recv_buf.data(),  // recvbuf
                                          2,                      // recvcount
                                          MPI_FLOAT,              // recvtype
                                          0,                      // root
                                          MPI_COMM_WORLD          // comm
                                          ),
                          "test_float",
                          float_expected_buf.data()  // expected_data
                          ),

          std::make_tuple(std::make_tuple(kDoubleData.data(),      // sendbuf
                                          2,                       // sendcount
                                          MPI_DOUBLE,              // sendtype
                                          double_recv_buf.data(),  // recvbuf
                                          2,                       // recvcount
                                          MPI_DOUBLE,              // recvtype
                                          0,                       // root
                                          MPI_COMM_WORLD           // comm
                                          ),
                          "test_double",
                          double_expected_buf.data()  // expected_data
                          ),

          std::make_tuple(std::make_tuple(nullptr,                // sendbuf
                                          0,                      // sendcount
                                          MPI_INT,                // sendtype
                                          empty_recv_buf.data(),  // recvbuf
                                          0,                      // recvcount
                                          MPI_INT,                // recvtype
                                          0,                      // root
                                          MPI_COMM_WORLD          // comm
                                          ),
                          "test_empty",
                          empty_expected_buf.data()  // expected_data
                          )};
}

const auto kTestParam = CreateTestParams();

TEST_P(PikhotskiyRScatterRunFuncTestsProcesses, ScatterTest) {
  ExecuteTest(GetParam());
}

const auto kTestTasksList = std::tuple_cat(
    ppc::util::AddFuncTask<PikhotskiyRScatterMPI, InType>(kTestParam, PPC_SETTINGS_pikhotskiy_r_scatter),
    ppc::util::AddFuncTask<PikhotskiyRScatterSEQ, InType>(kTestParam, PPC_SETTINGS_pikhotskiy_r_scatter));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName =
    PikhotskiyRScatterRunFuncTestsProcesses::PrintFuncTestName<PikhotskiyRScatterRunFuncTestsProcesses>;

INSTANTIATE_TEST_SUITE_P(ScatterTests, PikhotskiyRScatterRunFuncTestsProcesses, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace pikhotskiy_r_scatter
