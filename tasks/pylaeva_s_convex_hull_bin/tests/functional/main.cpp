#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <cstdlib>
#include <string>
#include <tuple>

#include "pylaeva_s_convex_hull_bin/common/include/common.hpp"
#include "pylaeva_s_convex_hull_bin/mpi/include/ops_mpi.hpp"
#include "pylaeva_s_convex_hull_bin/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"

namespace pylaeva_s_convex_hull_bin {

class PylaevaSConvexHullBinFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return std::to_string(std::get<0>(test_param)) + "_" + std::get<1>(test_param);
  }

 protected:
  void SetUp() override {}

  bool CheckTestOutputData(OutType &output_data) final {
    for (const auto &hull : output_data.convex_hulls) {
      if (hull.empty()) {
        return false;
      }

      for (size_t i = 0; i < hull.size(); ++i) {
        for (size_t j = i + 1; j < hull.size(); ++j) {
          if (hull[i] == hull[j]) {
            return false;
          }
        }
      }

      if (hull.size() == 1) {
        continue;
      }

      if (hull.size() == 2) {
        if (hull[0] == hull[1]) {
          return false;
        }
        continue;
      }

      if (hull.size() >= 3) {
        for (size_t i = 0; i < hull.size(); ++i) {
          const Point &p0 = hull[i];
          const Point &p1 = hull[(i + 1) % hull.size()];
          const Point &p2 = hull[(i + 2) % hull.size()];

          int cross = ((p1.x - p0.x) * (p2.y - p1.y)) - ((p1.y - p0.y) * (p2.x - p1.x));
          if (cross <= 0) {
            return false;
          }
        }
      }
    }
    return true;
  }

  InType GetTestInputData() final {
    InType data;

    auto test_param = std::get<2>(GetParam());
    int test_case = std::get<0>(test_param);

    data.width = 20 * test_case;
    data.height = 20 * test_case;
    data.pixels.resize(static_cast<size_t>(data.width) * static_cast<size_t>(data.height), 0);

    switch (test_case) {
      case 1:
        CreateSquarePattern(data);
        break;
      case 2:
        CreateTrianglePattern(data);
        break;
      case 3:
        CreateCirclePattern(data);
        break;
      case 4:
        CreateMultipleComponents(data);
        break;
      case 5:
        CreateLinePattern(data);
        break;
      case 6:
        CreatePatternWithHole(data);
        break;
      case 7:
        CreateLShapePattern(data);
        break;
      case 8:
        CreateSinglePoint(data);
        break;
      case 9:
        CreateTwoPoints(data);
        break;
      case 10:
        CreateColinearPoints(data);
        break;
      default:
        CreateSquarePattern(data);
    }

    return data;
  }

 private:
  static void CreateSquarePattern(ImageData &data) {
    int center_x = data.width / 2;
    int center_y = data.height / 2;
    int size = 6;

    for (int delta_y = -size / 2; delta_y <= size / 2; ++delta_y) {
      for (int delta_x = -size / 2; delta_x <= size / 2; ++delta_x) {
        int x = center_x + delta_x;
        int y = center_y + delta_y;
        if (x >= 0 && x < data.width && y >= 0 && y < data.height) {
          int idx = (y * data.width) + x;
          data.pixels[idx] = 255;
        }
      }
    }
  }

  static void CreateTrianglePattern(ImageData &data) {
    int center_x = data.width / 2;
    int center_y = data.height / 2;
    int size = 8;

    for (int row = 0; row < size; ++row) {
      for (int col = 0; col <= row; ++col) {
        int px = center_x - (size / 2) + col;
        int py = center_y - (size / 2) + row;
        if (px >= 0 && px < data.width && py >= 0 && py < data.height) {
          int idx = (py * data.width) + px;
          data.pixels[idx] = 255;
        }
      }
    }
  }

  static void CreateCirclePattern(ImageData &data) {
    int center_x = data.width / 2;
    int center_y = data.height / 2;
    int radius = 8;

    for (int delta_y = -radius; delta_y <= radius; ++delta_y) {
      for (int delta_x = -radius; delta_x <= radius; ++delta_x) {
        if ((delta_x * delta_x) + (delta_y * delta_y) <= radius * radius) {
          int px = center_x + delta_x;
          int py = center_y + delta_y;
          if (px >= 0 && px < data.width && py >= 0 && py < data.height) {
            int idx = (py * data.width) + px;
            data.pixels[idx] = 255;
          }
        }
      }
    }
  }

  static void CreateMultipleComponents(ImageData &data) {
    CreateSquarePattern(data);

    int small_size = 3;

    for (int row = 0; row < small_size; ++row) {
      for (int col = 0; col < small_size; ++col) {
        int idx = (row * data.width) + col;
        data.pixels[idx] = 255;
      }
    }

    for (int row = 0; row < small_size; ++row) {
      for (int col = data.width - small_size; col < data.width; ++col) {
        int idx = (row * data.width) + col;
        data.pixels[idx] = 255;
      }
    }

    for (int row = data.height - small_size; row < data.height; ++row) {
      for (int col = 0; col < small_size; ++col) {
        int idx = (row * data.width) + col;
        data.pixels[idx] = 255;
      }
    }
  }

  static void CreateLinePattern(ImageData &data) {
    int line_y = data.height / 2;
    for (int coord_x = data.width / 4; coord_x < 3 * data.width / 4; ++coord_x) {
      int idx = (line_y * data.width) + coord_x;
      data.pixels[idx] = 255;
    }

    int line_x = data.width / 2;
    for (int coord_y = data.height / 4; coord_y < 3 * data.height / 4; ++coord_y) {
      int idx = (coord_y * data.width) + line_x;
      data.pixels[idx] = 255;
    }
  }

  static void CreatePatternWithHole(ImageData &data) {
    int center_x = data.width / 2;
    int center_y = data.height / 2;
    int outer_size = 10;
    int inner_size = 4;

    for (int delta_y = -outer_size / 2; delta_y <= outer_size / 2; ++delta_y) {
      for (int delta_x = -outer_size / 2; delta_x <= outer_size / 2; ++delta_x) {
        if (std::abs(delta_x) <= inner_size / 2 && std::abs(delta_y) <= inner_size / 2) {
          continue;
        }
        int px = center_x + delta_x;
        int py = center_y + delta_y;
        if (px >= 0 && px < data.width && py >= 0 && py < data.height) {
          int idx = (py * data.width) + px;
          data.pixels[idx] = 255;
        }
      }
    }
  }

  static void CreateLShapePattern(ImageData &data) {
    int center_x = data.width / 2;
    int center_y = data.height / 2;
    int size = 8;

    for (int delta_y = -size / 2; delta_y <= size / 2; ++delta_y) {
      int x = center_x - (size / 2);
      int px = x;
      int py = center_y + delta_y;
      if (px >= 0 && px < data.width && py >= 0 && py < data.height) {
        int idx = (py * data.width) + px;
        data.pixels[idx] = 255;
      }
    }

    for (int delta_x = -size / 2; delta_x <= size / 2; ++delta_x) {
      int y_pos = center_y + (size / 2);
      int px = center_x + delta_x;
      int py = y_pos;
      if (px >= 0 && px < data.width && py >= 0 && py < data.height) {
        int idx = (py * data.width) + px;
        data.pixels[idx] = 255;
      }
    }
  }

  static void CreateSinglePoint(ImageData &data) {
    int center_x = data.width / 2;
    int center_y = data.height / 2;
    int idx = (center_y * data.width) + center_x;
    data.pixels[idx] = 255;
  }

  static void CreateTwoPoints(ImageData &data) {
    int center_x = data.width / 2;
    int center_y = data.height / 2;

    int idx1 = (center_y * data.width) + center_x;
    int idx2 = ((center_y + 5) * data.width) + center_x;

    data.pixels[idx1] = 255;
    data.pixels[idx2] = 255;
  }

  static void CreateColinearPoints(ImageData &data) {
    int center_x = data.width / 2;
    int center_y = data.height / 2;

    for (int i = -2; i <= 2; ++i) {
      int px = center_x + (i * 5);
      int py = center_y;
      if (px >= 0 && px < data.width) {
        int idx = (py * data.width) + px;
        data.pixels[idx] = 255;
      }
    }
  }
};

namespace {

TEST_P(PylaevaSConvexHullBinFuncTests, MatmulFromPic) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 10> kTestParam = {std::make_tuple(1, "square"),     std::make_tuple(2, "triangle"),
                                             std::make_tuple(3, "circle"),     std::make_tuple(4, "multiple"),
                                             std::make_tuple(5, "lines"),      std::make_tuple(6, "with_hole"),
                                             std::make_tuple(7, "L_shape"),    std::make_tuple(8, "single_point"),
                                             std::make_tuple(9, "two_points"), std::make_tuple(10, "colinear_points")};

const auto kTestTasksList = std::tuple_cat(
    ppc::util::AddFuncTask<PylaevaSConvexHullBinMPI, InType>(kTestParam, PPC_SETTINGS_pylaeva_s_convex_hull_bin),
    ppc::util::AddFuncTask<PylaevaSConvexHullBinSEQ, InType>(kTestParam, PPC_SETTINGS_pylaeva_s_convex_hull_bin));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName = PylaevaSConvexHullBinFuncTests::PrintFuncTestName<PylaevaSConvexHullBinFuncTests>;

INSTANTIATE_TEST_SUITE_P(PicMatrixTests, PylaevaSConvexHullBinFuncTests, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace pylaeva_s_convex_hull_bin
