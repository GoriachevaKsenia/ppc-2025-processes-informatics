#include "chyokotov_a_convex_hull_finding/seq/include/ops_seq.hpp"

#include <algorithm>
#include <array>
#include <climits>
#include <cstddef>
#include <queue>
#include <utility>
#include <vector>

#include "chyokotov_a_convex_hull_finding/common/include/common.hpp"

namespace chyokotov_a_convex_hull_finding {

ChyokotovConvexHullFindingSEQ::ChyokotovConvexHullFindingSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput().clear();
  GetInput().reserve(in.size());
  for (const auto &row : in) {
    GetInput().push_back(row);
  }

  GetOutput().clear();
}

bool ChyokotovConvexHullFindingSEQ::ValidationImpl() {
  const auto &input = GetInput();
  if (input.empty()) {
    return true;
  }

  for (size_t i = 0; i < input.size(); ++i) {
    for (size_t j = 0; j < input[0].size(); ++j) {
      if (input[i][j] != 1 && input[i][j] != 0) {
        return false;
      }
    }
  }

  size_t length_row = input[0].size();
  return std::ranges::all_of(input, [length_row](const auto &row) { return row.size() == length_row; });
}

bool ChyokotovConvexHullFindingSEQ::PreProcessingImpl() {
  GetOutput().clear();
  return true;
}

std::vector<std::pair<int, int>> ChyokotovConvexHullFindingSEQ::Bfs(int start_x, int start_y,
                                                                    const std::vector<std::vector<int>> &picture,
                                                                    std::vector<std::vector<bool>> &visited) {
  std::vector<std::pair<int, int>> component;
  std::queue<std::pair<int, int>> queue;

  const std::array<std::pair<int, int>, 4> directions = {{{0, -1}, {0, 1}, {-1, 0}, {1, 0}}};

  queue.emplace(start_x, start_y);
  visited[start_y][start_x] = true;

  while (!queue.empty()) {
    auto [current_x, current_y] = queue.front();
    queue.pop();

    component.emplace_back(current_x, current_y);

    for (const auto &[dx, dy] : directions) {
      int neighbor_x = current_x + dx;
      int neighbor_y = current_y + dy;

      if (neighbor_x >= 0 && std::cmp_less(neighbor_x, static_cast<int>(picture[0].size())) && neighbor_y >= 0 &&
          std::cmp_less(neighbor_y, static_cast<int>(picture.size()))) {
        if (picture[neighbor_y][neighbor_x] == 1 && !visited[neighbor_y][neighbor_x]) {
          visited[neighbor_y][neighbor_x] = true;
          queue.emplace(neighbor_x, neighbor_y);
        }
      }
    }
  }

  return component;
}

std::vector<std::vector<std::pair<int, int>>> ChyokotovConvexHullFindingSEQ::FindComponent() {
  auto picture = GetInput();
  int rows = static_cast<int>(picture.size());
  int cols = static_cast<int>(picture[0].size());

  std::vector<std::vector<std::pair<int, int>>> components;
  std::vector<std::vector<bool>> visited(rows, std::vector<bool>(cols, false));

  for (int row_idx = 0; row_idx < rows; ++row_idx) {
    for (int col_idx = 0; col_idx < cols; ++col_idx) {
      if (picture[row_idx][col_idx] == 1 && !visited[row_idx][col_idx]) {
        auto component = Bfs(col_idx, row_idx, picture, visited);
        components.emplace_back(std::move(component));
      }
    }
  }

  return components;
}

int ChyokotovConvexHullFindingSEQ::Cross(const std::pair<int, int> &o, const std::pair<int, int> &a,
                                         const std::pair<int, int> &b) {
  return ((a.first - o.first) * (b.second - o.second)) - ((a.second - o.second) * (b.first - o.first));
}

std::vector<std::pair<int, int>> ChyokotovConvexHullFindingSEQ::ConvexHull(std::vector<std::pair<int, int>> x) {
  int n = static_cast<int>(x.size());

  if (n <= 2) {
    return x;
  }

  std::ranges::sort(x.begin(), x.end());

  std::vector<std::pair<int, int>> hull(2 * static_cast<size_t>(n));
  int k = 0;

  for (int i = 0; i < n; i++) {
    while (k >= 2 && Cross(hull[k - 2], hull[k - 1], x[i]) <= 0) {
      k--;
    }
    hull[k++] = x[i];
  }

  for (int i = n - 2, tk = k + 1; i >= 0; i--) {
    while (k >= tk && Cross(hull[k - 2], hull[k - 1], x[i]) <= 0) {
      k--;
    }
    hull[k++] = x[i];
  }

  hull.resize(k - 1);
  return hull;
}

bool ChyokotovConvexHullFindingSEQ::RunImpl() {
  auto &picture = GetInput();
  if (picture.empty()) {
    return true;
  }
  auto &output = GetOutput();
  auto components = FindComponent();
  for (auto &i : components) {
    output.push_back(ConvexHull(i));
  }

  return true;
}

bool ChyokotovConvexHullFindingSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace chyokotov_a_convex_hull_finding
