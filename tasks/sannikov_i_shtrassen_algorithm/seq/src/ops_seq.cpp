#include "sannikov_i_shtrassen_algorithm/seq/include/ops_seq.hpp"

#include <cstddef>
#include <tuple>
#include <utility>
#include <vector>

#include "sannikov_i_shtrassen_algorithm/common/include/common.hpp"

namespace sannikov_i_shtrassen_algorithm {

namespace {

using Matrix = std::vector<std::vector<double>>;
using Flat = std::vector<double>;

constexpr std::size_t kClassicThreshold = 2;

std::size_t NextPow2(std::size_t value) {
  std::size_t pow2 = 1;
  while (pow2 < value) {
    pow2 <<= 1U;
  }
  return pow2;
}

std::size_t Idx(std::size_t row, std::size_t col, std::size_t ld) {
  return (row * ld) + col;
}

struct View {
  const double *data = nullptr;
  std::size_t ld = 0;
  std::size_t row0 = 0;
  std::size_t col0 = 0;
  std::size_t n = 0;
};

View MakeView(const Flat &buf, std::size_t n) {
  View v;
  v.data = buf.data();
  v.ld = n;
  v.row0 = 0;
  v.col0 = 0;
  v.n = n;
  return v;
}

View SubView(const View &parent, std::size_t dr, std::size_t dc, std::size_t n) {
  View v;
  v.data = parent.data;
  v.ld = parent.ld;
  v.row0 = parent.row0 + dr;
  v.col0 = parent.col0 + dc;
  v.n = n;
  return v;
}

double At(const View &v, std::size_t req, std::size_t ceq) {
  return v.data[Idx(v.row0 + req, v.col0 + ceq, v.ld)];
}

void PackPadToFlat(const Matrix &src, std::size_t n0, std::size_t m, Flat *out_flat) {
  out_flat->assign(m * m, 0.0);
  for (std::size_t row = 0; row < n0; ++row) {
    for (std::size_t col = 0; col < n0; ++col) {
      (*out_flat)[Idx(row, col, m)] = src[row][col];
    }
  }
}

Matrix UnpackCropFromFlat(const Flat &flat, std::size_t m, std::size_t n0) {
  Matrix out(n0, std::vector<double>(n0, 0.0));
  for (std::size_t row = 0; row < n0; ++row) {
    for (std::size_t col = 0; col < n0; ++col) {
      out[row][col] = flat[Idx(row, col, m)];
    }
  }
  return out;
}

struct OwnedView {
  Flat buf;
  View view;
};

void FillAdd(const View &a, const View &b, Flat *out) {
  const std::size_t n = a.n;
  out->assign(n * n, 0.0);
  for (std::size_t req = 0; req < n; ++req) {
    for (std::size_t ceq = 0; ceq < n; ++ceq) {
      (*out)[Idx(req, ceq, n)] = At(a, req, ceq) + At(b, req, ceq);
    }
  }
}

void FillSub(const View &a, const View &b, Flat *out) {
  const std::size_t n = a.n;
  out->assign(n * n, 0.0);
  for (std::size_t req = 0; req < n; ++req) {
    for (std::size_t ceq = 0; ceq < n; ++ceq) {
      (*out)[Idx(req, ceq, n)] = At(a, req, ceq) - At(b, req, ceq);
    }
  }
}

Flat MultiplyClassicView(const View &a, const View &b) {
  const std::size_t n = a.n;
  Flat ceq(n * n, 0.0);

  for (std::size_t req = 0; req < n; ++req) {
    double *crow = &ceq[Idx(req, 0, n)];
    for (std::size_t k = 0; k < n; ++k) {
      const double aik = At(a, req, k);
      for (std::size_t col = 0; col < n; ++col) {
        crow[col] += aik * At(b, k, col);
      }
    }
  }
  return ceq;
}

void PlaceBlockFlat(const Flat &blk, std::size_t blk_n, Flat *dst, std::size_t dst_n, std::size_t row0,
                    std::size_t col0) {
  for (std::size_t req = 0; req < blk_n; ++req) {
    for (std::size_t ceq = 0; ceq < blk_n; ++ceq) {
      (*dst)[Idx(row0 + req, col0 + ceq, dst_n)] = blk[Idx(req, ceq, blk_n)];
    }
  }
}

struct Frame {
  OwnedView a_owned;
  OwnedView b_owned;
  View a;
  View b;
  std::size_t n = 0;
  bool has_parent = false;
  std::size_t parent_idx = 0;
  int parent_slot = 0;
  int stage = 0;
  Flat res;
  bool split_ready = false;
  View a11, a12, a21, a22;
  View b11, b12, b21, b22;

  Flat m1, m2, m3, m4, m5, m6, m7;
};

bool IsLeaf(const Frame &f) {
  return f.n <= kClassicThreshold;
}

void EnsureSplit(Frame *f) {
  if (f->split_ready) {
    return;
  }
  const std::size_t half = f->n / 2;
  f->a11 = SubView(f->a, 0, 0, half);
  f->a12 = SubView(f->a, 0, half, half);
  f->a21 = SubView(f->a, half, 0, half);
  f->a22 = SubView(f->a, half, half, half);

  f->b11 = SubView(f->b, 0, 0, half);
  f->b12 = SubView(f->b, 0, half, half);
  f->b21 = SubView(f->b, half, 0, half);
  f->b22 = SubView(f->b, half, half, half);

  f->split_ready = true;
  f->stage = 1;
}

void AssignToParent(std::vector<Frame> *stack, const Frame &child) {
  if (!child.has_parent) {
    return;
  }

  Frame &parent = (*stack)[child.parent_idx];
  const int slot = child.parent_slot;

  if (slot == 1) {
    parent.m1 = child.res;
  } else if (slot == 2) {
    parent.m2 = child.res;
  } else if (slot == 3) {
    parent.m3 = child.res;
  } else if (slot == 4) {
    parent.m4 = child.res;
  } else if (slot == 5) {
    parent.m5 = child.res;
  } else if (slot == 6) {
    parent.m6 = child.res;
  } else {
    parent.m7 = child.res;
  }
}

Frame MakeChild(const Frame &parent, std::size_t parent_idx, int slot) {
  const std::size_t half = parent.n / 2;

  Frame child;
  child.has_parent = true;
  child.parent_idx = parent_idx;
  child.parent_slot = slot;
  child.stage = 0;
  child.split_ready = false;
  child.n = half;
  child.a_owned.buf.clear();
  child.b_owned.buf.clear();
  auto set_from_buf = [&](Flat *buf, OwnedView *ov) {
    ov->buf = std::move(*buf);
    ov->view.data = ov->buf.data();
    ov->view.ld = half;
    ov->view.row0 = 0;
    ov->view.col0 = 0;
    ov->view.n = half;
  };

  if (slot == 1) {
    Flat tmp_a;
    Flat tmp_b;
    FillAdd(parent.a11, parent.a22, &tmp_a);
    FillAdd(parent.b11, parent.b22, &tmp_b);
    set_from_buf(&tmp_a, &child.a_owned);
    set_from_buf(&tmp_b, &child.b_owned);
    child.a = child.a_owned.view;
    child.b = child.b_owned.view;
    return child;
  }

  if (slot == 2) {
    Flat tmp_a;
    FillAdd(parent.a21, parent.a22, &tmp_a);
    set_from_buf(&tmp_a, &child.a_owned);
    child.a = child.a_owned.view;
    child.b = parent.b11;
    return child;
  }

  if (slot == 3) {
    Flat tmp_b;
    FillSub(parent.b12, parent.b22, &tmp_b);
    set_from_buf(&tmp_b, &child.b_owned);
    child.a = parent.a11;
    child.b = child.b_owned.view;
    return child;
  }

  if (slot == 4) {
    Flat tmp_b;
    FillSub(parent.b21, parent.b11, &tmp_b);
    set_from_buf(&tmp_b, &child.b_owned);
    child.a = parent.a22;
    child.b = child.b_owned.view;
    return child;
  }
  if (slot == 5) {
    Flat tmp_a;
    FillAdd(parent.a11, parent.a12, &tmp_a);
    set_from_buf(&tmp_a, &child.a_owned);
    child.a = child.a_owned.view;
    child.b = parent.b22;
    return child;
  }
  if (slot == 6) {
    Flat tmp_a;
    Flat tmp_b;
    FillSub(parent.a21, parent.a11, &tmp_a);
    FillAdd(parent.b11, parent.b12, &tmp_b);
    set_from_buf(&tmp_a, &child.a_owned);
    set_from_buf(&tmp_b, &child.b_owned);
    child.a = child.a_owned.view;
    child.b = child.b_owned.view;
    return child;
  }
  {
    Flat tmp_a;
    Flat tmp_b;
    FillSub(parent.a12, parent.a22, &tmp_a);
    FillAdd(parent.b21, parent.b22, &tmp_b);
    set_from_buf(&tmp_a, &child.a_owned);
    set_from_buf(&tmp_b, &child.b_owned);
    child.a = child.a_owned.view;
    child.b = child.b_owned.view;
    return child;
  }
}

void Combine(Frame *f) {
  const std::size_t half = f->n / 2;
  const std::size_t kk = half * half;

  Flat c11(kk, 0.0);
  Flat c12(kk, 0.0);
  Flat c21(kk, 0.0);
  Flat c22(kk, 0.0);

  for (std::size_t i = 0; i < kk; ++i) {
    c11[i] = f->m1[i] + f->m4[i] - f->m5[i] + f->m7[i];
  }
  for (std::size_t i = 0; i < kk; ++i) {
    c12[i] = f->m3[i] + f->m5[i];
  }
  for (std::size_t i = 0; i < kk; ++i) {
    c21[i] = f->m2[i] + f->m4[i];
  }
  for (std::size_t i = 0; i < kk; ++i) {
    c22[i] = f->m1[i] - f->m2[i] + f->m3[i] + f->m6[i];
  }

  f->res.assign(f->n * f->n, 0.0);
  PlaceBlockFlat(c11, half, &f->res, f->n, 0, 0);
  PlaceBlockFlat(c12, half, &f->res, f->n, 0, half);
  PlaceBlockFlat(c21, half, &f->res, f->n, half, 0);
  PlaceBlockFlat(c22, half, &f->res, f->n, half, half);
}

Flat Shtrassen(const View &a0, const View &b0) {
  std::vector<Frame> stack;
  stack.reserve(64);

  Frame root;
  root.a = a0;
  root.b = b0;
  root.n = a0.n;
  root.has_parent = false;
  root.stage = 0;
  root.split_ready = false;
  stack.push_back(std::move(root));

  while (!stack.empty()) {
    Frame &cur = stack.back();

    if (IsLeaf(cur)) {
      cur.res = MultiplyClassicView(cur.a, cur.b);
      const Frame finished = cur;
      stack.pop_back();

      if (stack.empty()) {
        return finished.res;
      }

      AssignToParent(&stack, finished);
      continue;
    }

    EnsureSplit(&cur);

    if (cur.stage >= 1 && cur.stage <= 7) {
      const int slot = cur.stage;
      const std::size_t parent_idx = stack.size() - 1;

      Frame child = MakeChild(cur, parent_idx, slot);
      cur.stage += 1;
      stack.push_back(std::move(child));
      continue;
    }

    if (cur.stage != 8) {
      cur.stage = 8;
      continue;
    }

    Combine(&cur);

    const Frame finished = cur;
    stack.pop_back();

    if (stack.empty()) {
      return finished.res;
    }

    AssignToParent(&stack, finished);
  }

  return Flat{};
}

}  // namespace

SannikovIShtrassenAlgorithmSEQ::SannikovIShtrassenAlgorithmSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  auto &input_buffer = GetInput();
  InType tmp(in);
  input_buffer.swap(tmp);
  GetOutput().clear();
}

bool SannikovIShtrassenAlgorithmSEQ::ValidationImpl() {
  (void)this;
  const auto &input = GetInput();
  const auto &mat_a = std::get<0>(input);
  const auto &mat_b = std::get<1>(input);

  if (mat_a.empty() || mat_b.empty()) {
    return false;
  }
  if (mat_a.size() != mat_b.size()) {
    return false;
  }
  if (mat_a.front().empty() || mat_b.front().empty()) {
    return false;
  }

  const auto n = mat_a.size();
  for (const auto &row : mat_a) {
    if (row.size() != n) {
      return false;
    }
  }
  for (const auto &row : mat_b) {
    if (row.size() != n) {
      return false;
    }
  }

  return GetOutput().empty();
}

bool SannikovIShtrassenAlgorithmSEQ::PreProcessingImpl() {
  (void)this;
  GetOutput().clear();
  return true;
}

bool SannikovIShtrassenAlgorithmSEQ::RunImpl() {
  (void)this;
  const auto &input = GetInput();
  const auto &a_in = std::get<0>(input);
  const auto &b_in = std::get<1>(input);

  const std::size_t n0 = a_in.size();
  const std::size_t m = NextPow2(n0);
  Flat a_flat;
  Flat b_flat;
  PackPadToFlat(a_in, n0, m, &a_flat);
  PackPadToFlat(b_in, n0, m, &b_flat);
  const View a0 = MakeView(a_flat, m);
  const View b0 = MakeView(b_flat, m);
  const Flat c_flat = Shtrassen(a0, b0);
  GetOutput() = UnpackCropFromFlat(c_flat, m, n0);

  return !GetOutput().empty();
}

bool SannikovIShtrassenAlgorithmSEQ::PostProcessingImpl() {
  (void)this;
  return !GetOutput().empty();
}

}  // namespace sannikov_i_shtrassen_algorithm
