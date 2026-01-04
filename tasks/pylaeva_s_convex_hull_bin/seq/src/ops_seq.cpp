#include "pylaeva_s_convex_hull_bin/seq/include/ops_seq.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <queue>
#include <utility>
#include <vector>

#include "pylaeva_s_convex_hull_bin/common/include/common.hpp"

namespace pylaeva_s_convex_hull_bin {

namespace {

int Cross(const Point &o, const Point &a, const Point &b) {
  return ((a.x - o.x) * (b.y - o.y)) - ((a.y - o.y) * (b.x - o.x));
}

void ProcessPixelNeighbors(const Point &p, int width, int height, const ImageData &processed_data,
                           std::vector<bool> &visited, std::queue<Point> &q) {
  const std::vector<std::pair<int, int>> directions = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};

  for (const auto &dir : directions) {
    int nx = p.x + dir.first;
    int ny = p.y + dir.second;

    if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
      int nidx = (ny * width) + nx;
      if (processed_data.pixels[static_cast<size_t>(nidx)] == 255 && !visited[static_cast<size_t>(nidx)]) {
        visited[static_cast<size_t>(nidx)] = true;
        q.emplace(nx, ny);
      }
    }
  }
}

void ProcessConnectedComponent(int start_x, int start_y, int width, int height, const ImageData &processed_data,
                               std::vector<bool> &visited, std::vector<std::vector<Point>> &components) {
  std::vector<Point> component;
  std::queue<Point> q;
  size_t start_idx = (static_cast<size_t>(start_y) * static_cast<size_t>(width)) + static_cast<size_t>(start_x);
  q.emplace(start_x, start_y);
  visited[start_idx] = true;

  while (!q.empty()) {
    Point p = q.front();
    q.pop();
    component.push_back(p);

    ProcessPixelNeighbors(p, width, height, processed_data, visited, q);
  }

  if (!component.empty()) {
    components.push_back(component);
  }
}

}  // namespace

PylaevaSConvexHullBinSEQ::PylaevaSConvexHullBinSEQ(const InType &in) : processed_data_(in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

bool PylaevaSConvexHullBinSEQ::ValidationImpl() {
  return GetInput().width > 0 && GetInput().height > 0 && !GetInput().pixels.empty() &&
         GetInput().pixels.size() == static_cast<size_t>(GetInput().width) * static_cast<size_t>(GetInput().height);
}

bool PylaevaSConvexHullBinSEQ::PreProcessingImpl() {
  const uint8_t threshold = 128;
  for (auto &pixel : processed_data_.pixels) {
    pixel = (pixel > threshold) ? 255 : 0;
  }
  return true;
}

bool PylaevaSConvexHullBinSEQ::RunImpl() {
  FindConnectedComponents();
  processed_data_.convex_hulls.clear();

  for (const auto &component : processed_data_.components) {
    if (component.size() >= 3) {
      processed_data_.convex_hulls.push_back(GrahamScan(component));
    } else if (!component.empty()) {
      processed_data_.convex_hulls.push_back(component);
    }
  }

  GetOutput() = processed_data_;
  return true;
}

bool PylaevaSConvexHullBinSEQ::PostProcessingImpl() {
  return true;
}

void PylaevaSConvexHullBinSEQ::FindConnectedComponents() {
  int width = processed_data_.width;
  int height = processed_data_.height;
  int total_pixels = width * height;
  std::vector<bool> visited(static_cast<size_t>(total_pixels), false);
  processed_data_.components.clear();

  for (int row_y = 0; row_y < height; ++row_y) {
    for (int col_x = 0; col_x < width; ++col_x) {
      size_t idx = (static_cast<size_t>(row_y) * static_cast<size_t>(width)) + static_cast<size_t>(col_x);
      if (processed_data_.pixels[idx] == 255 && !visited[idx]) {
        ProcessConnectedComponent(col_x, row_y, width, height, processed_data_, visited, processed_data_.components);
      }
    }
  }
}

std::vector<Point> PylaevaSConvexHullBinSEQ::GrahamScan(const std::vector<Point> &points) {
  if (points.size() <= 3) {
    return points;
  }

  std::vector<Point> pts = points;
  size_t n = pts.size();

  size_t min_idx = 0;
  for (size_t i = 1; i < n; ++i) {
    if (pts[i].y < pts[min_idx].y || (pts[i].y == pts[min_idx].y && pts[i].x < pts[min_idx].x)) {
      min_idx = i;
    }
  }
  std::swap(pts[0], pts[min_idx]);

  Point pivot = pts[0];
  std::sort(pts.begin() + 1, pts.end(), [&pivot](const Point &a, const Point &b) {
    int orient = Cross(pivot, a, b);
    if (orient == 0) {
      return ((a.x - pivot.x) * (a.x - pivot.x)) + ((a.y - pivot.y) * (a.y - pivot.y)) <
             ((b.x - pivot.x) * (b.x - pivot.x)) + ((b.y - pivot.y) * (b.y - pivot.y));
    }
    return orient > 0;
  });

  std::vector<Point> hull;
  for (size_t i = 0; i < n; ++i) {
    while (hull.size() >= 2 && Cross(hull[hull.size() - 2], hull.back(), pts[i]) <= 0) {
      hull.pop_back();
    }
    hull.push_back(pts[i]);
  }

  return hull;
}
}  // namespace pylaeva_s_convex_hull_bin
