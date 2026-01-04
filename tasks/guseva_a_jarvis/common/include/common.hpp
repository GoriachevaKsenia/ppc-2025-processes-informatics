#pragma once

#include <cstddef>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "task/include/task.hpp"

namespace guseva_a_jarvis {

using InType = std::tuple<int, int, std::vector<int>>;
using OutType = std::vector<int>;
using TestFromFileType = std::tuple<int, int, std::vector<int>, std::vector<int>>;
using TestType = std::string;
using BaseTask = ppc::task::Task<InType, OutType>;

inline TestFromFileType ReadTestDataFromFile(const std::string &filename) {
  std::ifstream file(filename);
  if (!file.is_open()) {
    throw std::runtime_error("Cannot open file: " + filename);
  }
  int width = 0;
  int height = 0;
  std::vector<int> input;
  std::vector<int> output;
  file >> width >> height;
  std::string line;
  std::getline(file, line);
  std::getline(file, line);
  std::stringstream ss(line);
  int num = 0;
  while (ss >> num) {
    input.push_back(num);
  }
  std::getline(file, line);
  ss.clear();
  ss.str(line);
  while (ss >> num) {
    output.push_back(num);
  }
  file.close();
  return std::make_tuple(width, height, input, output);
}

inline int CrossProduct(const std::pair<int, int> &a, const std::pair<int, int> &b, const std::pair<int, int> &c) {
  return ((b.first - a.first) * (c.second - a.second)) - ((b.second - a.second) * (c.first - a.first));
}

inline int DistanceSquared(const std::pair<int, int> &a, const std::pair<int, int> &b) {
  int dx = b.first - a.first;
  int dy = b.second - a.second;
  return (dx * dx) + (dy * dy);
}

inline size_t FindLeftmostBottommostPoint(const std::vector<std::pair<int, int>> &points) {
  size_t start_idx = 0;
  for (size_t i = 1; i < points.size(); ++i) {
    if (points[i].first < points[start_idx].first ||
        (points[i].first == points[start_idx].first && points[i].second < points[start_idx].second)) {
      start_idx = i;
    }
  }
  return start_idx;
}

inline size_t FindNextHullPoint(const std::vector<std::pair<int, int>> &points, size_t current_idx,
                                size_t candidate_idx) {
  size_t next_idx = candidate_idx;

  for (size_t i = 0; i < points.size(); ++i) {
    if (i == current_idx) {
      continue;
    }

    const std::pair<int, int> &current_point = points[current_idx];
    const std::pair<int, int> &next_point = points[next_idx];
    const std::pair<int, int> &check_point = points[i];

    int cross = CrossProduct(current_point, next_point, check_point);

    if (cross < 0) {
      next_idx = i;
    } else if (cross == 0) {
      int dist_to_next = DistanceSquared(current_point, next_point);
      int dist_to_check = DistanceSquared(current_point, check_point);
      if (dist_to_check > dist_to_next) {
        next_idx = i;
      }
    }
  }

  return next_idx;
}

inline std::vector<std::pair<int, int>> BuildConvexHull(const std::vector<std::pair<int, int>> &points) {
  if (points.size() <= 3) {
    return points;
  }

  std::vector<std::pair<int, int>> hull;

  const size_t start_idx = FindLeftmostBottommostPoint(points);
  size_t current_idx = start_idx;
  bool returned_to_start = false;

  hull.reserve(points.size());

  while (!returned_to_start) {
    hull.push_back(points[current_idx]);

    size_t next_idx = (current_idx + 1) % points.size();
    next_idx = FindNextHullPoint(points, current_idx, next_idx);

    if (next_idx == start_idx) {
      returned_to_start = true;
    } else {
      current_idx = next_idx;
    }
  }

  return hull;
}

}  // namespace guseva_a_jarvis
