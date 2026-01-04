#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <set>

#include "pylaeva_s_convex_hull_bin/common/include/common.hpp"
#include "pylaeva_s_convex_hull_bin/mpi/include/ops_mpi.hpp"
#include "pylaeva_s_convex_hull_bin/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace pylaeva_s_convex_hull_bin {

#ifndef M_PI
#  define M_PI 3.14159265358979323846
#endif

class PylaevaSConvexHullBinPerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
 protected:
  void SetUp() override {}

  bool CheckTestOutputData(OutType &output_data) final {
    if (output_data.width <= 0 || output_data.height <= 0) {
      return false;
    }

    for (const auto &hull : output_data.convex_hulls) {
      if (hull.empty()) {
        return false;
      }
      std::set<Point> unique_points;
      for (const auto &point : hull) {
        if (!unique_points.insert(point).second) {
          return false;
        }
      }

      for (const auto &point : hull) {
        if (point.x < 0 || point.x >= output_data.width || point.y < 0 || point.y >= output_data.height) {
          return false;
        }
      }

      if (hull.size() >= 3) {
        bool all_positive = true;
        bool all_negative = true;

        for (size_t i = 0; i < hull.size(); ++i) {
          const Point &p0 = hull[i];
          const Point &p1 = hull[(i + 1) % hull.size()];
          const Point &p2 = hull[(i + 2) % hull.size()];

          int cross = ((p1.x - p0.x) * (p2.y - p1.y)) - ((p1.y - p0.y) * (p2.x - p1.x));

          if (cross > 0) {
            all_negative = false;
          }
          if (cross < 0) {
            all_positive = false;
          }
        }

        if (!(all_positive || all_negative)) {
          return false;
        }
      }
      if (hull.size() == 2) {
        if (hull[0] == hull[1]) {
          return false;
        }
      }
    }
    return true;
  }

  InType GetTestInputData() final {
    InType data;

    int base_size = 1000;

    data.width = base_size;
    data.height = base_size;
    data.pixels.resize(static_cast<size_t>(data.width) * static_cast<size_t>(data.height), 0);

    CreateHeavyTestPattern(data);

    return data;
  }

 private:
  static void DrawMainCircle(ImageData &data, int center_x, int center_y, int big_radius) {
    for (int coord_y = -big_radius; coord_y <= big_radius; ++coord_y) {
      for (int coord_x = -big_radius; coord_x <= big_radius; ++coord_x) {
        if ((coord_x * coord_x) + (coord_y * coord_y) <= big_radius * big_radius) {
          int px = center_x + coord_x;
          int py = center_y + coord_y;
          if (px >= 0 && px < data.width && py >= 0 && py < data.height) {
            size_t idx = (static_cast<size_t>(py) * static_cast<size_t>(data.width)) + static_cast<size_t>(px);
            data.pixels[idx] = 255;
          }
        }
      }
    }
  }

  static void DrawSmallCircles(ImageData &data, int center_x, int center_y, int big_radius) {
    int small_radius = data.width / 20;
    int circle_count = 16;

    for (int index = 0; index < circle_count; ++index) {
      double angle = 2.0 * M_PI * index / circle_count;
      int circ_x = center_x + static_cast<int>(big_radius * 1.7 * std::cos(angle));
      int circ_y = center_y + static_cast<int>(big_radius * 1.7 * std::sin(angle));

      for (int coord_y = -small_radius; coord_y <= small_radius; ++coord_y) {
        for (int coord_x = -small_radius; coord_x <= small_radius; ++coord_x) {
          if ((coord_x * coord_x) + (coord_y * coord_y) <= small_radius * small_radius) {
            int px = circ_x + coord_x;
            int py = circ_y + coord_y;
            if (px >= 0 && px < data.width && py >= 0 && py < data.height) {
              size_t idx = (static_cast<size_t>(py) * static_cast<size_t>(data.width)) + static_cast<size_t>(px);
              data.pixels[idx] = 255;
            }
          }
        }
      }
    }
  }

  static void DrawHorizontalGridLines(ImageData &data) {
    int grid_cells = 20;
    int cell_height = data.height / grid_cells;

    for (int index = 1; index < grid_cells; ++index) {
      int line_y = index * cell_height;
      int line_thickness = 2;

      for (int coord_y = line_y - line_thickness; coord_y <= line_y + line_thickness; ++coord_y) {
        if (coord_y >= 0 && coord_y < data.height) {
          for (int coord_x = 0; coord_x < data.width; ++coord_x) {
            size_t idx =
                (static_cast<size_t>(coord_y) * static_cast<size_t>(data.width)) + static_cast<size_t>(coord_x);
            data.pixels[idx] = 255;
          }
        }
      }
    }
  }

  static void DrawVerticalGridLines(ImageData &data) {
    int grid_cells = 20;
    int cell_width = data.width / grid_cells;

    for (int index = 1; index < grid_cells; ++index) {
      int line_x = index * cell_width;
      int line_thickness = 2;

      for (int coord_x = line_x - line_thickness; coord_x <= line_x + line_thickness; ++coord_x) {
        if (coord_x >= 0 && coord_x < data.width) {
          for (int coord_y = 0; coord_y < data.height; ++coord_y) {
            if (coord_y % 4 < 2) {
              size_t idx =
                  (static_cast<size_t>(coord_y) * static_cast<size_t>(data.width)) + static_cast<size_t>(coord_x);
              data.pixels[idx] = 255;
            }
          }
        }
      }
    }
  }

  static void DrawRectangles(ImageData &data) {
    int rect_count = 8;
    for (int index = 0; index < rect_count; ++index) {
      int rect_width = data.width / (rect_count / 2 + 2);
      int rect_height = data.height / (rect_count / 2 + 2);

      int x1 = (index * 2 * data.width) / rect_count;
      int y1 = (index * 2 * data.height) / rect_count;
      int x2 = x1 + (rect_width * (index % 2 + 1));
      int y2 = y1 + (rect_height * (index % 3 + 1));

      x1 = std::max(0, x1);
      y1 = std::max(0, y1);
      x2 = std::min(data.width - 1, x2);
      y2 = std::min(data.height - 1, y2);

      for (int coord_y = y1; coord_y <= y2; ++coord_y) {
        for (int coord_x = x1; coord_x <= x2; ++coord_x) {
          size_t idx = (static_cast<size_t>(coord_y) * static_cast<size_t>(data.width)) + static_cast<size_t>(coord_x);
          data.pixels[idx] = 255;
        }
      }
    }
  }

  static void DrawDiagonalLines(ImageData &data) {
    int diagonal_count = 15;
    for (int index = 0; index < diagonal_count; ++index) {
      int start_x = (index * data.width) / diagonal_count;
      int start_y = 0;
      int end_x = data.width;
      int end_y = (index * data.height) / diagonal_count;

      DrawThickLine(data, start_x, start_y, end_x, end_y, 3);
    }

    for (int index = 0; index < diagonal_count; ++index) {
      int start_x = 0;
      int start_y = (index * data.height) / diagonal_count;
      int end_x = data.width;
      int end_y = (index * data.height) / diagonal_count;

      DrawThickLine(data, start_x, start_y, end_x, end_y, 3);
    }
  }

  static void CreateHeavyTestPattern(ImageData &data) {
    std::ranges::fill(data.pixels, 0);

    int center_x = data.width / 2;
    int center_y = data.height / 2;
    int big_radius = std::min(data.width, data.height) / 3;

    DrawMainCircle(data, center_x, center_y, big_radius);
    DrawSmallCircles(data, center_x, center_y, big_radius);
    DrawHorizontalGridLines(data);
    DrawVerticalGridLines(data);
    DrawRectangles(data);
    DrawDiagonalLines(data);
  }

  static void DrawThickLine(ImageData &data, int x1, int y1, int x2, int y2, int thickness) {
    for (int thickness_offset = -thickness / 2; thickness_offset <= thickness / 2; ++thickness_offset) {
      DrawLine(data, x1, y1 + thickness_offset, x2, y2 + thickness_offset);
      DrawLine(data, x1 + thickness_offset, y1, x2 + thickness_offset, y2);
    }
  }

  static void DrawLine(ImageData &data, int x1, int y1, int x2, int y2) {
    int dx = std::abs(x2 - x1);
    int dy = std::abs(y2 - y1);
    int sx = (x1 < x2) ? 1 : -1;
    int sy = (y1 < y2) ? 1 : -1;
    int err = dx - dy;

    while (true) {
      if (x1 >= 0 && x1 < data.width && y1 >= 0 && y1 < data.height) {
        size_t idx = (static_cast<size_t>(y1) * static_cast<size_t>(data.width)) + static_cast<size_t>(x1);
        data.pixels[idx] = 255;
      }

      if (x1 == x2 && y1 == y2) {
        break;
      }

      int e2 = 2 * err;
      if (e2 > -dy) {
        err -= dy;
        x1 += sx;
      }
      if (e2 < dx) {
        err += dx;
        y1 += sy;
      }
    }
  }
};

TEST_P(PylaevaSConvexHullBinPerfTests, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks = ppc::util::MakeAllPerfTasks<InType, PylaevaSConvexHullBinSEQ, PylaevaSConvexHullBinMPI>(
    PPC_SETTINGS_pylaeva_s_convex_hull_bin);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = PylaevaSConvexHullBinPerfTests::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, PylaevaSConvexHullBinPerfTests, kGtestValues, kPerfTestName);

}  // namespace pylaeva_s_convex_hull_bin
