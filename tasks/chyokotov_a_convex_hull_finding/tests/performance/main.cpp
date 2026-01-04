#include <gtest/gtest.h>

#include <climits>
#include <cstddef>
#include <set>
#include <utility>
#include <vector>

#include "chyokotov_a_convex_hull_finding/common/include/common.hpp"
#include "chyokotov_a_convex_hull_finding/mpi/include/ops_mpi.hpp"
#include "chyokotov_a_convex_hull_finding/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace chyokotov_a_convex_hull_finding {

class ChyokotovConvexHullPerfTest : public ppc::util::BaseRunPerfTests<InType, OutType> {
  InType input_data_;
  OutType expected_;

  void SetUp() override {
    const int height = 8000;
    const int width = 8000;
    input_data_.resize(height);
    for (auto &row : input_data_) {
      row.resize(width, 0);
    }
    expected_.clear();

    int rect_top = 1000;
    int rect_left = 1000;
    int rect_height = 100;
    int rect_width = 1000;

    for (int yt = rect_top; yt < rect_top + rect_height; ++yt) {
      for (int xt = rect_left; xt < rect_left + rect_width; ++xt) {
        input_data_[yt][xt] = 1;
      }
    }
    expected_.push_back({{1000, 1000}, {1999, 1000}, {1999, 1099}, {1000, 1099}});

    int cross_center_y = 4500;
    int cross_center_x = 4500;
    int cross_size = 50;

    for (int xt = cross_center_x - cross_size; xt <= cross_center_x + cross_size; ++xt) {
      input_data_[cross_center_y][xt] = 1;
    }

    for (int yt = cross_center_y - 50; yt <= cross_center_y + 50; ++yt) {
      input_data_[yt][cross_center_x] = 1;
    }
    expected_.push_back({{4450, 4500}, {4550, 4500}, {4500, 4550}, {4500, 4450}});

    int frame_top = 3000;
    int frame_left = 200;
    int frame_size = 100;

    for (int xt = frame_left; xt < frame_left + frame_size; ++xt) {
      input_data_[frame_top][xt] = 1;
      input_data_[frame_top + frame_size - 1][xt] = 1;
    }

    for (int yt = frame_top; yt < frame_top + frame_size; ++yt) {
      input_data_[yt][frame_left] = 1;
      input_data_[yt][frame_left + frame_size - 1] = 1;
    }
    expected_.push_back({{200, 3000}, {299, 3000}, {299, 3099}, {200, 3099}});

    int small_component_count = 100;

    for (int comp = 0; comp < small_component_count; ++comp) {
      int y = comp * 50;
      int x = 100 + (comp * 40);

      if (comp % 4 == 0) {
        input_data_[y][x] = input_data_[y][x + 1] = 1;
        input_data_[y + 1][x] = input_data_[y + 1][x + 1] = 1;
        expected_.push_back({{x, y}, {x + 1, y}, {x + 1, y + 1}, {x, y + 1}});
      } else if (comp % 4 == 1) {
        input_data_[y][x] = input_data_[y][x + 1] = input_data_[y][x + 2] = 1;
        expected_.push_back({{x, y}, {x + 2, y}});
      } else if (comp % 4 == 2) {
        input_data_[y][x] = input_data_[y + 1][x] = input_data_[y + 2][x] = 1;
        expected_.push_back({{x, y}, {x, y + 2}});
      } else {
        input_data_[y][x] = input_data_[y][x + 1] = input_data_[y][x + 2] = 1;
        input_data_[y + 1][x] = input_data_[y + 1][x + 1] = input_data_[y + 1][x + 2] = 1;
        expected_.push_back({{x, y}, {x + 2, y}, {x + 2, y + 1}, {x, y + 1}});
      }
    }
    int chess_start_y = 6000;
    int chess_start_x = 6000;

    for (int yt = chess_start_y; yt < chess_start_y + 500; ++yt) {
      for (int xt = chess_start_x; xt < chess_start_x + 100; ++xt) {
        if (((yt - chess_start_y) + (xt - chess_start_x)) % 2 == 0) {
          input_data_[yt][xt] = 1;
          expected_.push_back({{xt, yt}});
        }
      }
    }
  }

  bool CheckTestOutputData(OutType &output_data) final {
    size_t points_actual = 0;
    size_t points_expected = 0;

    for (const auto &i : output_data) {
      points_actual += i.size();
    }

    for (const auto &i : expected_) {
      points_expected += i.size();
    }

    if (points_actual != points_expected) {
      return false;
    }

    std::multiset<std::pair<int, int>> set_actual;
    std::multiset<std::pair<int, int>> set_expected;

    for (const auto &i : output_data) {
      set_actual.insert(i.begin(), i.end());
    }

    for (const auto &i : expected_) {
      set_expected.insert(i.begin(), i.end());
    }

    return (set_actual == set_expected);
  }

  InType GetTestInputData() final {
    return input_data_;
  }
};

TEST_P(ChyokotovConvexHullPerfTest, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, ChyokotovConvexHullFindingMPI, ChyokotovConvexHullFindingSEQ>(
        PPC_SETTINGS_chyokotov_a_convex_hull_finding);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = ChyokotovConvexHullPerfTest::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, ChyokotovConvexHullPerfTest, kGtestValues, kPerfTestName);

}  // namespace chyokotov_a_convex_hull_finding
