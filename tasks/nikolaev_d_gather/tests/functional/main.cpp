#include <gtest/gtest.h>
#include <mpi.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <string>
#include <tuple>
#include <vector>

#include "nikolaev_d_gather/common/include/common.hpp"
#include "nikolaev_d_gather/mpi/include/ops_mpi.hpp"
#include "nikolaev_d_gather/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"

namespace nikolaev_d_gather {

class NikolaevDGatherPerfTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    const auto &count = std::get<0>(test_param);
    const auto &root = std::get<1>(test_param);
    const auto &datatype = std::get<2>(test_param);
    const auto &name = std::get<3>(test_param);

    std::string type_str;
    if (datatype == MPI_INT) {
      type_str = "int";
    } else if (datatype == MPI_FLOAT) {
      type_str = "float";
    } else if (datatype == MPI_DOUBLE) {
      type_str = "double";
    }

    return "count" + std::to_string(count) + "_root" + std::to_string(root) + "_" + type_str + "_" + name;
  }

 protected:
  void SetUp() override {
    TestType params = std::get<2>(GetParam());

    const auto &count = std::get<0>(params);
    const auto &root = std::get<1>(params);
    const auto &datatype = std::get<2>(params);

    int type_size = 0;
    if (datatype == MPI_INT) {
      type_size = sizeof(int);
    } else if (datatype == MPI_FLOAT) {
      type_size = sizeof(float);
    } else if (datatype == MPI_DOUBLE) {
      type_size = sizeof(double);
    }

    size_t total_size = static_cast<size_t>(count) * static_cast<size_t>(type_size);
    if (total_size == 0) {
      total_size = 1;
    }
    std::vector<char> data(total_size);

    if (datatype == MPI_INT) {
      auto *data_ptr = reinterpret_cast<int *>(data.data());
      for (int i = 0; i < count; ++i) {
        data_ptr[i] = i + 1;
      }
    } else if (datatype == MPI_FLOAT) {
      auto *data_ptr = reinterpret_cast<float *>(data.data());
      for (int i = 0; i < count; ++i) {
        data_ptr[i] = static_cast<float>(i) + 0.5F;
      }
    } else if (datatype == MPI_DOUBLE) {
      auto *data_ptr = reinterpret_cast<double *>(data.data());
      for (int i = 0; i < count; ++i) {
        data_ptr[i] = static_cast<double>(i) + 0.25;
      }
    }

    input_data_ = GatherInput{.data = data, .count = count, .datatype = datatype, .root = root};
  }

  static int GetTypeSize(MPI_Datatype datatype) {
    if (datatype == MPI_INT) {
      return sizeof(int);
    }
    if (datatype == MPI_FLOAT) {
      return sizeof(float);
    }
    if (datatype == MPI_DOUBLE) {
      return sizeof(double);
    }
    return 0;
  }

  bool CheckTestOutputData(OutType &output_data) final {
    const auto &input = input_data_;
    std::string test_name = std::get<1>(GetParam());
    bool is_mpi = test_name.find("_mpi_") != std::string::npos;

    int size = 1;
    if (is_mpi) {
      MPI_Comm_size(MPI_COMM_WORLD, &size);
      int rank = 0;
      MPI_Comm_rank(MPI_COMM_WORLD, &rank);
      if (rank != input.root) {
        return true;
      }
    }

    if (output_data.empty()) {
      return false;
    }

    const auto type_size = static_cast<size_t>(GetTypeSize(input.datatype));
    const size_t expected_size = static_cast<size_t>(input.count) * static_cast<size_t>(size) * type_size;

    return output_data.size() == expected_size;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
};

namespace {

TEST_P(NikolaevDGatherPerfTests, GatherCheck) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 3> kTestParam = {std::make_tuple(3, 0, MPI_INT, "SmallIntRoot0"),
                                            std::make_tuple(4, 0, MPI_FLOAT, "SmallFloatRoot0"),
                                            std::make_tuple(3, 0, MPI_DOUBLE, "SmallDoubleRoot0")};

const auto kTestTasksList =
    std::tuple_cat(ppc::util::AddFuncTask<NikolaevDGatherMPI, InType>(kTestParam, PPC_SETTINGS_nikolaev_d_gather),
                   ppc::util::AddFuncTask<NikolaevDGatherSEQ, InType>(kTestParam, PPC_SETTINGS_nikolaev_d_gather));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName = NikolaevDGatherPerfTests::PrintFuncTestName<NikolaevDGatherPerfTests>;

INSTANTIATE_TEST_SUITE_P(GatherTests, NikolaevDGatherPerfTests, kGtestValues, kPerfTestName);

TEST(NikolaevDGatherMPITest, BasicMPIGather) {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  std::vector<char> data(3 * sizeof(int));
  auto *data_ptr = reinterpret_cast<int *>(data.data());
  data_ptr[0] = 1;
  data_ptr[1] = 2;
  data_ptr[2] = 3;

  GatherInput input{.data = data, .count = 3, .datatype = MPI_INT, .root = 0};
  NikolaevDGatherMPI task(input);

  EXPECT_TRUE(task.Validation());
  EXPECT_TRUE(task.PreProcessing());
  EXPECT_TRUE(task.Run());
  EXPECT_TRUE(task.PostProcessing());

  if (rank == 0) {
    const auto &result = task.GetOutput();
    EXPECT_FALSE(result.empty());
  }
}

TEST(NikolaevDGatherSEQTest, BasicSEQGather) {
  std::vector<char> data(3 * sizeof(int));
  auto *data_ptr = reinterpret_cast<int *>(data.data());
  data_ptr[0] = 1;
  data_ptr[1] = 2;
  data_ptr[2] = 3;

  GatherInput input{.data = data, .count = 3, .datatype = MPI_INT, .root = 0};
  NikolaevDGatherSEQ task(input);

  EXPECT_TRUE(task.Validation());
  EXPECT_TRUE(task.PreProcessing());
  EXPECT_TRUE(task.Run());
  EXPECT_TRUE(task.PostProcessing());

  const auto &result = task.GetOutput();
  ASSERT_EQ(result.size(), 3 * sizeof(int));

  const int *res_ptr = reinterpret_cast<const int *>(result.data());
  EXPECT_EQ(res_ptr[0], 1);
  EXPECT_EQ(res_ptr[1], 2);
  EXPECT_EQ(res_ptr[2], 3);
}

TEST(NikolaevDGatherMPITest, InvalidValidation) {
  std::vector<char> data(sizeof(int));
  GatherInput input_neg_count{.data = data, .count = -1, .datatype = MPI_INT, .root = 0};
  NikolaevDGatherMPI task_1(input_neg_count);
  EXPECT_FALSE(task_1.Validation());

  int size = 1;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  GatherInput input_invalid_root{.data = data, .count = 1, .datatype = MPI_INT, .root = size + 1};
  NikolaevDGatherMPI task_2(input_invalid_root);
  EXPECT_FALSE(task_2.Validation());

  GatherInput input_wrong_type{.data = data, .count = 1, .datatype = MPI_CHAR, .root = 0};
  NikolaevDGatherMPI task_3(input_wrong_type);
  EXPECT_FALSE(task_3.Validation());
}

TEST(NikolaevDGatherMPITest, MiddleRootGather) {
  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  int root = size / 2;
  int count = 2;
  std::vector<char> data(static_cast<size_t>(count) * sizeof(float));
  auto *d_ptr = reinterpret_cast<float *>(data.data());
  for (int i = 0; i < count; ++i) {
    d_ptr[i] = static_cast<float>(rank);
  }

  GatherInput input{.data = data, .count = count, .datatype = MPI_FLOAT, .root = root};
  NikolaevDGatherMPI task(input);

  ASSERT_TRUE(task.Validation());
  task.PreProcessing();
  task.Run();
  task.PostProcessing();

  if (rank == root) {
    const auto &result = task.GetOutput();
    ASSERT_EQ(result.size(), static_cast<size_t>(count) * static_cast<size_t>(size) * sizeof(float));
    const auto *res_ptr = reinterpret_cast<const float *>(result.data());
    for (int rank_id = 0; rank_id < size; ++rank_id) {
      for (int i = 0; i < count; ++i) {
        EXPECT_FLOAT_EQ(res_ptr[(rank_id * count) + i], static_cast<float>(rank_id));
      }
    }
  }
}

TEST(NikolaevDGatherMPITest, LargeDataGather) {
  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  int count = 10000;
  std::vector<char> data(count * sizeof(double));
  auto *d_ptr = reinterpret_cast<double *>(data.data());
  std::fill(d_ptr, d_ptr + count, static_cast<double>(rank));

  GatherInput input{.data = data, .count = count, .datatype = MPI_DOUBLE, .root = 0};
  NikolaevDGatherMPI task(input);

  task.Validation();
  task.PreProcessing();
  task.Run();
  task.PostProcessing();

  if (rank == 0) {
    const auto &result = task.GetOutput();
    EXPECT_EQ(result.size(), static_cast<size_t>(count) * static_cast<size_t>(size) * sizeof(double));
  }
}

}  // namespace

}  // namespace nikolaev_d_gather
