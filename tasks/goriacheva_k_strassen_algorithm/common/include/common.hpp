#pragma once

#include <cstddef>
#include <string>
#include <tuple>
#include <vector>

#include "task/include/task.hpp"

namespace goriacheva_k_strassen_algorithm {

using Matrix = std::vector<std::vector<double>>;

struct InType {
  Matrix a;
  Matrix b;
};

using OutType = Matrix;
using TestType = std::tuple<InType, OutType, std::string>;
using BaseTask = ppc::task::Task<InType, OutType>;

inline std::vector<double> Flatten(const Matrix &m) {
  std::vector<double> buf;
  buf.reserve(m.size() * m.size());
  for (const auto &row : m) {
    buf.insert(buf.end(), row.begin(), row.end());
  }
  return buf;
}

inline Matrix UnFlatten(const std::vector<double> &buf, std::size_t n) {
  Matrix m(n, std::vector<double>(n));
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t j = 0; j < n; ++j) {
      m[i][j] = buf[(i * n) + j];
    }
  }
  return m;
}

inline bool IsSquare(const Matrix &m) {
  if (m.empty()) {
    return false;
  }
  std::size_t n = m.size();
  // NOLINT(readability-use-anyofallof)
  for (const auto &r : m) {
    if (r.size() != n) {
      return false;
    }
  }
  return true;
  // NOLINT(readability-use-anyofallof)
}

inline std::size_t NextPowerOfTwo(std::size_t n) {
  std::size_t p = 1;
  while (p < n) {
    p <<= 1;
  }
  return p;
}

inline Matrix PadMatrix(const Matrix &a, std::size_t new_size) {
  Matrix r(new_size, std::vector<double>(new_size, 0.0));
  for (std::size_t i = 0; i < a.size(); ++i) {
    for (std::size_t j = 0; j < a.size(); ++j) {
      r[i][j] = a[i][j];
    }
  }
  return r;
}

inline Matrix CropMatrix(const Matrix &a, std::size_t size) {
  Matrix r(size, std::vector<double>(size));
  for (std::size_t i = 0; i < size; ++i) {
    for (std::size_t j = 0; j < size; ++j) {
      r[i][j] = a[i][j];
    }
  }
  return r;
}

inline Matrix Add(const Matrix &a, const Matrix &b) {
  std::size_t n = a.size();
  Matrix c(n, std::vector<double>(n));
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t j = 0; j < n; ++j) {
      c[i][j] = a[i][j] + b[i][j];
    }
  }
  return c;
}

inline Matrix Sub(const Matrix &a, const Matrix &b) {
  std::size_t n = a.size();
  Matrix c(n, std::vector<double>(n));
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t j = 0; j < n; ++j) {
      c[i][j] = a[i][j] - b[i][j];
    }
  }
  return c;
}

inline Matrix NaiveMultiply(const Matrix &a, const Matrix &b) {
  std::size_t n = a.size();
  Matrix c(n, std::vector<double>(n, 0.0));
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t k = 0; k < n; ++k) {
      for (std::size_t j = 0; j < n; ++j) {
        c[i][j] += a[i][k] * b[k][j];
      }
    }
  }
  return c;
}

// NOLINTBEGIN(misc-no-recursion)
inline Matrix Strassen(const Matrix &a, const Matrix &b, std::size_t threshold = 128) {
  std::size_t n = a.size();
  if (n <= threshold) {
    return NaiveMultiply(a, b);
  }

  std::size_t k = n / 2;

  Matrix a11(k, std::vector<double>(k));
  Matrix a12(k, std::vector<double>(k));
  Matrix a21(k, std::vector<double>(k));
  Matrix a22(k, std::vector<double>(k));
  Matrix b11(k, std::vector<double>(k));
  Matrix b12(k, std::vector<double>(k));
  Matrix b21(k, std::vector<double>(k));
  Matrix b22(k, std::vector<double>(k));

  for (std::size_t i = 0; i < k; ++i) {
    for (std::size_t j = 0; j < k; ++j) {
      a11[i][j] = a[i][j];
      a12[i][j] = a[i][j + k];
      a21[i][j] = a[i + k][j];
      a22[i][j] = a[i + k][j + k];
      b11[i][j] = b[i][j];
      b12[i][j] = b[i][j + k];
      b21[i][j] = b[i + k][j];
      b22[i][j] = b[i + k][j + k];
    }
  }

  Matrix m1 = Strassen(Add(a11, a22), Add(b11, b22), threshold);
  Matrix m2 = Strassen(Add(a21, a22), b11, threshold);
  Matrix m3 = Strassen(a11, Sub(b12, b22), threshold);
  Matrix m4 = Strassen(a22, Sub(b21, b11), threshold);
  Matrix m5 = Strassen(Add(a11, a12), b22, threshold);
  Matrix m6 = Strassen(Sub(a21, a11), Add(b11, b12), threshold);
  Matrix m7 = Strassen(Sub(a12, a22), Add(b21, b22), threshold);

  Matrix c(n, std::vector<double>(n));
  for (std::size_t i = 0; i < k; ++i) {
    for (std::size_t j = 0; j < k; ++j) {
      c[i][j] = m1[i][j] + m4[i][j] - m5[i][j] + m7[i][j];
      c[i][j + k] = m3[i][j] + m5[i][j];
      c[i + k][j] = m2[i][j] + m4[i][j];
      c[i + k][j + k] = m1[i][j] - m2[i][j] + m3[i][j] + m6[i][j];
    }
  }

  return c;
}
// NOLINTBEGIN(misc-no-recursion)

}  // namespace goriacheva_k_strassen_algorithm
