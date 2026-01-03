#include <gtest/gtest.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "sizov_d_global_search/common/include/common.hpp"
#include "sizov_d_global_search/mpi/include/ops_mpi.hpp"
#include "sizov_d_global_search/seq/include/ops_seq.hpp"
#include "task/include/task.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace sizov_d_global_search {

namespace {

enum class TokenType : std::uint8_t {
  kEnd,
  kNumber,
  kIdentifier,
  kPlus,
  kMinus,
  kStar,
  kSlash,
  kCaret,
  kLParen,
  kRParen
};

struct Token {
  TokenType type = TokenType::kEnd;
  std::string text;
  double number_value = 0.0;
};

class Lexer {
 public:
  explicit Lexer(std::string expression) : expression_(std::move(expression)) {}

  Token Next() {
    SkipWhitespace();
    if (pos_ >= expression_.size()) {
      Token t;
      t.type = TokenType::kEnd;
      return t;
    }

    const char ch = expression_[pos_];
    if ((std::isdigit(static_cast<unsigned char>(ch)) != 0) || ch == '.') {
      return ParseNumber();
    }
    if (IsIdentifierStart(ch)) {
      return ParseIdentifier();
    }

    ++pos_;
    Token t;
    if (ch == '+') {
      t.type = TokenType::kPlus;
      t.text = "+";
      return t;
    }
    if (ch == '-') {
      t.type = TokenType::kMinus;
      t.text = "-";
      return t;
    }
    if (ch == '*') {
      t.type = TokenType::kStar;
      t.text = "*";
      return t;
    }
    if (ch == '/') {
      t.type = TokenType::kSlash;
      t.text = "/";
      return t;
    }
    if (ch == '^') {
      t.type = TokenType::kCaret;
      t.text = "^";
      return t;
    }
    if (ch == '(') {
      t.type = TokenType::kLParen;
      t.text = "(";
      return t;
    }
    if (ch == ')') {
      t.type = TokenType::kRParen;
      t.text = ")";
      return t;
    }

    throw std::runtime_error("Unexpected character");
  }

 private:
  void SkipWhitespace() {
    while (pos_ < expression_.size() && (std::isspace(static_cast<unsigned char>(expression_[pos_])) != 0)) {
      ++pos_;
    }
  }

  Token ParseNumber() {
    const char *begin = expression_.c_str() + pos_;
    char *end = nullptr;
    const double value = std::strtod(begin, &end);
    if (begin == end) {
      throw std::runtime_error("Invalid number");
    }

    pos_ = static_cast<std::size_t>(end - expression_.c_str());

    Token t;
    t.type = TokenType::kNumber;
    t.number_value = value;
    return t;
  }

  Token ParseIdentifier() {
    const std::size_t start = pos_;
    while (pos_ < expression_.size() && IsIdentifierChar(expression_[pos_])) {
      ++pos_;
    }

    Token token;
    token.type = TokenType::kIdentifier;
    token.text = expression_.substr(start, pos_ - start);
    return token;
  }

  static bool IsIdentifierStart(char ch) {
    return (std::isalpha(static_cast<unsigned char>(ch)) != 0) || ch == '_';
  }

  static bool IsIdentifierChar(char ch) {
    return (std::isalnum(static_cast<unsigned char>(ch)) != 0) || ch == '_';
  }

  std::string expression_;
  std::size_t pos_ = 0;
};

class Expr {
 public:
  virtual ~Expr() = default;
  [[nodiscard]] virtual double Eval(double x) const = 0;
};

using ExprPtr = std::unique_ptr<Expr>;

class ConstantExpr : public Expr {
 public:
  explicit ConstantExpr(double v) : v_(v) {}

  [[nodiscard]] double Eval(double x) const override {
    (void)x;
    return v_;
  }

 private:
  double v_;
};

class VariableExpr : public Expr {
 public:
  [[nodiscard]] double Eval(double x) const override {
    return x;
  }
};

class UnaryExpr : public Expr {
 public:
  UnaryExpr(char op, ExprPtr child) : op_(op), child_(std::move(child)) {}

  [[nodiscard]] double Eval(double x) const override {
    const double v = child_->Eval(x);
    if (op_ == '-') {
      return -v;
    }
    return v;
  }

 private:
  char op_;
  ExprPtr child_;
};

class BinaryExpr : public Expr {
 public:
  BinaryExpr(char op, ExprPtr l, ExprPtr r) : op_(op), left_(std::move(l)), right_(std::move(r)) {}

  [[nodiscard]] double Eval(double x) const override {
    const double a = left_->Eval(x);
    const double b = right_->Eval(x);
    switch (op_) {
      case '+':
        return a + b;
      case '-':
        return a - b;
      case '*':
        return a * b;
      case '/':
        return a / b;
      case '^':
        return std::pow(a, b);
      default:
        break;
    }
    return std::numeric_limits<double>::quiet_NaN();
  }

 private:
  char op_;
  ExprPtr left_;
  ExprPtr right_;
};

class FunctionCallExpr : public Expr {
 public:
  using FuncPtr = double (*)(double);

  FunctionCallExpr(FuncPtr f, ExprPtr arg) : func_(f), arg_(std::move(arg)) {}

  [[nodiscard]] double Eval(double x) const override {
    return func_(arg_->Eval(x));
  }

 private:
  FuncPtr func_;
  ExprPtr arg_;
};

class Parser {
 public:
  explicit Parser(Lexer lex) : lexer_(std::move(lex)) {
    Advance();
  }

  ExprPtr Parse() {
    ExprPtr expr = ParseExpression();
    Expect(TokenType::kEnd);
    return expr;
  }

 private:
  struct OperatorEntry {
    enum class Kind : std::uint8_t { kPlus, kMinus, kMul, kDiv, kPow, kUnaryMinus, kLParen, kFunc };
    Kind kind;
    FunctionCallExpr::FuncPtr func = nullptr;
  };

  void Advance() {
    current_ = lexer_.Next();
  }

  [[nodiscard]] bool Check(TokenType t) const {
    return current_.type == t;
  }

  void Expect(TokenType t) {
    if (!Check(t)) {
      throw std::runtime_error("Unexpected token");
    }
    Advance();
  }

  static int Precedence(const OperatorEntry &op) {
    switch (op.kind) {
      case OperatorEntry::Kind::kPlus:
      case OperatorEntry::Kind::kMinus:
        return 1;
      case OperatorEntry::Kind::kMul:
      case OperatorEntry::Kind::kDiv:
        return 2;
      case OperatorEntry::Kind::kPow:
        return 3;
      case OperatorEntry::Kind::kUnaryMinus:
      case OperatorEntry::Kind::kFunc:
        return 4;
      case OperatorEntry::Kind::kLParen:
        return 0;
    }
    return 0;
  }

  static bool IsRightAssociative(const OperatorEntry &op) {
    return op.kind == OperatorEntry::Kind::kPow || op.kind == OperatorEntry::Kind::kUnaryMinus;
  }

  static void ApplyOperator(std::vector<ExprPtr> &values, const OperatorEntry &op) {
    switch (op.kind) {
      case OperatorEntry::Kind::kUnaryMinus: {
        auto a = std::move(values.back());
        values.pop_back();
        values.push_back(std::make_unique<UnaryExpr>('-', std::move(a)));
        return;
      }
      case OperatorEntry::Kind::kFunc: {
        auto arg = std::move(values.back());
        values.pop_back();
        values.push_back(std::make_unique<FunctionCallExpr>(op.func, std::move(arg)));
        return;
      }
      case OperatorEntry::Kind::kPlus:
      case OperatorEntry::Kind::kMinus:
      case OperatorEntry::Kind::kMul:
      case OperatorEntry::Kind::kDiv:
      case OperatorEntry::Kind::kPow: {
        auto rhs = std::move(values.back());
        values.pop_back();
        auto lhs = std::move(values.back());
        values.pop_back();

        char c = '?';
        if (op.kind == OperatorEntry::Kind::kPlus) {
          c = '+';
        } else if (op.kind == OperatorEntry::Kind::kMinus) {
          c = '-';
        } else if (op.kind == OperatorEntry::Kind::kMul) {
          c = '*';
        } else if (op.kind == OperatorEntry::Kind::kDiv) {
          c = '/';
        } else if (op.kind == OperatorEntry::Kind::kPow) {
          c = '^';
        }

        values.push_back(std::make_unique<BinaryExpr>(c, std::move(lhs), std::move(rhs)));
        return;
      }
      case OperatorEntry::Kind::kLParen:
        return;
    }
  }

  static FunctionCallExpr::FuncPtr ResolveFunction(const std::string &name) {
    if (name == "sin") {
      return std::sin;
    }
    if (name == "cos") {
      return std::cos;
    }
    if (name == "tan") {
      return std::tan;
    }
    if (name == "exp") {
      return std::exp;
    }
    if (name == "log") {
      return std::log;
    }
    if (name == "sqrt") {
      return std::sqrt;
    }
    if (name == "abs") {
      return static_cast<double (*)(double)>(std::fabs);
    }
    throw std::runtime_error("Unknown function");
  }

  static void PushUnaryMinus(std::vector<OperatorEntry> &ops) {
    ops.push_back({OperatorEntry::Kind::kUnaryMinus, nullptr});
  }

  static void PushLParen(std::vector<OperatorEntry> &ops) {
    ops.push_back({OperatorEntry::Kind::kLParen, nullptr});
  }

  static void PushFunction(std::vector<OperatorEntry> &ops, const std::string &id) {
    ops.push_back({OperatorEntry::Kind::kFunc, ResolveFunction(id)});
    PushLParen(ops);
  }

  static void PushBinaryOperator(std::vector<ExprPtr> &values, std::vector<OperatorEntry> &ops,
                                 OperatorEntry::Kind kind) {
    OperatorEntry new_op{.kind = kind, .func = nullptr};
    while (!ops.empty() && ops.back().kind != OperatorEntry::Kind::kLParen) {
      const int top = Precedence(ops.back());
      const int now = Precedence(new_op);
      if (top > now || (!IsRightAssociative(new_op) && top == now)) {
        ApplyOperator(values, ops.back());
        ops.pop_back();
      } else {
        break;
      }
    }
    ops.push_back(new_op);
  }

  void HandleIdentifierToken(std::vector<ExprPtr> &values, std::vector<OperatorEntry> &ops, bool &expect_value) {
    std::string id = current_.text;
    Advance();

    if (Check(TokenType::kLParen)) {
      PushFunction(ops, id);
      Advance();
      expect_value = true;
      return;
    }

    if (id == "x") {
      values.push_back(std::make_unique<VariableExpr>());
      expect_value = false;
      return;
    }
    if (id == "pi") {
      values.push_back(std::make_unique<ConstantExpr>(std::acos(-1.0)));
      expect_value = false;
      return;
    }

    throw std::runtime_error("Unknown identifier");
  }

  void HandleExpectValue(std::vector<ExprPtr> &values, std::vector<OperatorEntry> &ops, bool &expect_value) {
    if (Check(TokenType::kNumber)) {
      const double v = current_.number_value;
      Advance();
      values.push_back(std::make_unique<ConstantExpr>(v));
      expect_value = false;
      return;
    }
    if (Check(TokenType::kIdentifier)) {
      HandleIdentifierToken(values, ops, expect_value);
      return;
    }
    if (Check(TokenType::kLParen)) {
      PushLParen(ops);
      Advance();
      expect_value = true;
      return;
    }
    if (Check(TokenType::kPlus) || Check(TokenType::kMinus)) {
      if (Check(TokenType::kMinus)) {
        PushUnaryMinus(ops);
      }
      Advance();
      expect_value = true;
      return;
    }

    throw std::runtime_error("Expected value");
  }

  static void HandleRParen(std::vector<ExprPtr> &values, std::vector<OperatorEntry> &ops) {
    while (!ops.empty() && ops.back().kind != OperatorEntry::Kind::kLParen) {
      ApplyOperator(values, ops.back());
      ops.pop_back();
    }
    if (ops.empty()) {
      throw std::runtime_error("Mismatched parentheses");
    }
    ops.pop_back();

    if (!ops.empty() && ops.back().kind == OperatorEntry::Kind::kFunc) {
      ApplyOperator(values, ops.back());
      ops.pop_back();
    }
  }

  bool HandleOperatorOrEnd(std::vector<ExprPtr> &values, std::vector<OperatorEntry> &ops, bool &expect_value) {
    if (Check(TokenType::kPlus) || Check(TokenType::kMinus) || Check(TokenType::kStar) || Check(TokenType::kSlash) ||
        Check(TokenType::kCaret)) {
      OperatorEntry::Kind kind = OperatorEntry::Kind::kPlus;
      if (Check(TokenType::kPlus)) {
        kind = OperatorEntry::Kind::kPlus;
      } else if (Check(TokenType::kMinus)) {
        kind = OperatorEntry::Kind::kMinus;
      } else if (Check(TokenType::kStar)) {
        kind = OperatorEntry::Kind::kMul;
      } else if (Check(TokenType::kSlash)) {
        kind = OperatorEntry::Kind::kDiv;
      } else {
        kind = OperatorEntry::Kind::kPow;
      }

      PushBinaryOperator(values, ops, kind);
      Advance();
      expect_value = true;
      return true;
    }

    if (Check(TokenType::kRParen)) {
      HandleRParen(values, ops);
      Advance();
      return true;
    }

    if (Check(TokenType::kEnd)) {
      return false;
    }

    throw std::runtime_error("Unexpected token");
  }

  static void FinalizeExpression(std::vector<ExprPtr> &values, std::vector<OperatorEntry> &ops) {
    while (!ops.empty()) {
      if (ops.back().kind == OperatorEntry::Kind::kLParen) {
        throw std::runtime_error("Mismatched parentheses");
      }
      ApplyOperator(values, ops.back());
      ops.pop_back();
    }
    if (values.size() != 1) {
      throw std::runtime_error("Invalid expression");
    }
  }

  ExprPtr ParseExpression() {
    std::vector<ExprPtr> values;
    std::vector<OperatorEntry> ops;
    values.reserve(16);
    ops.reserve(16);

    bool expect_value = true;
    while (true) {
      if (expect_value) {
        HandleExpectValue(values, ops, expect_value);
      } else {
        if (!HandleOperatorOrEnd(values, ops, expect_value)) {
          break;
        }
      }
    }

    FinalizeExpression(values, ops);
    return std::move(values.back());
  }

  Lexer lexer_;
  Token current_{};
};

struct SharedExprFunctor {
  std::shared_ptr<Expr> ptr;

  double operator()(double x) const {
    return ptr->Eval(x);
  }
};

Function BuildFunction(const nlohmann::json &json) {
  const std::string expr = json.at("expression");
  Lexer lex(expr);
  Parser parser(std::move(lex));
  ExprPtr ast = parser.Parse();
  auto sp = std::shared_ptr<Expr>(std::move(ast));
  return Function{SharedExprFunctor{sp}};
}

struct ExpectedSolution {
  std::vector<double> argmins;
  double value = 0.0;
};

struct LocalTestCase {
  std::string name;
  Problem problem;
  ExpectedSolution expected;
};

using LocalTestType = LocalTestCase;
using FuncParam = ppc::util::FuncTestParam<InType, OutType, LocalTestType>;

std::vector<LocalTestType> LoadTestCasesFromData() {
  namespace fs = std::filesystem;

  fs::path json_path = ppc::util::GetAbsoluteTaskPath(PPC_ID_sizov_d_global_search, "tests.json");
  std::ifstream file(json_path);
  if (!file) {
    throw std::runtime_error("Cannot open tests.json");
  }

  nlohmann::json data;
  file >> data;
  if (!data.is_array()) {
    throw std::runtime_error("tests.json must contain array");
  }

  std::vector<LocalTestType> cases;
  cases.reserve(data.size());

  for (const auto &item : data) {
    LocalTestType t{};
    t.name = item.at("name").get<std::string>();

    const auto &pj = item.at("problem");

    Problem p{};
    p.left = pj.at("left").get<double>();
    p.right = pj.at("right").get<double>();

    // defaults
    p.accuracy = 1e-4;
    p.reliability = 3.0;
    p.max_iterations = 300;

    // optional per-test settings
    if (item.contains("settings")) {
      const auto &s = item.at("settings");

      if (s.contains("accuracy")) {
        p.accuracy = s.at("accuracy").get<double>();
      }
      if (s.contains("reliability")) {
        p.reliability = s.at("reliability").get<double>();
      }
      if (s.contains("max_iterations")) {
        p.max_iterations = s.at("max_iterations").get<int>();
      }
    }

    p.func = BuildFunction(item.at("function"));
    t.problem = std::move(p);

    const auto &ej = item.at("expected");
    t.expected.argmins = ej.at("argmins").get<std::vector<double>>();
    t.expected.value = ej.at("value").get<double>();

    cases.push_back(std::move(t));
  }

  return cases;
}

std::vector<FuncParam> BuildTestTasks(const std::vector<LocalTestType> &tests) {
  std::vector<FuncParam> tasks;
  tasks.reserve(tests.size() * 2U);

  std::string mpi_name =
      std::string(ppc::util::GetNamespace<SizovDGlobalSearchMPI>()) + "_" +
      ppc::task::GetStringTaskType(SizovDGlobalSearchMPI::GetStaticTypeOfTask(), PPC_SETTINGS_sizov_d_global_search);
  std::string seq_name =
      std::string(ppc::util::GetNamespace<SizovDGlobalSearchSEQ>()) + "_" +
      ppc::task::GetStringTaskType(SizovDGlobalSearchSEQ::GetStaticTypeOfTask(), PPC_SETTINGS_sizov_d_global_search);

  for (const auto &t : tests) {
    tasks.emplace_back(ppc::task::TaskGetter<SizovDGlobalSearchMPI, InType>, mpi_name, t);
    tasks.emplace_back(ppc::task::TaskGetter<SizovDGlobalSearchSEQ, InType>, seq_name, t);
  }

  return tasks;
}

}  // namespace

class SizovDGlobalSearchFunctionalTests : public ppc::util::BaseRunFuncTests<InType, OutType, LocalTestType> {
 public:
  void RunFunctionalTestCase(const FuncParam &param) {
    PrepareTestCase(param);
    ExecuteTest(param);
  }

  void TestBody() override {}

  static std::string PrintTestParam(const LocalTestType &t) {
    return t.name;
  }

  bool CheckTestOutputData(OutType &o) final {
    if (!std::isfinite(o.value)) {
      return false;
    }

    const double dv = std::abs(o.value - expected_.value);
    const double dv_tol = 20.0 * input_.accuracy;

    if (dv > dv_tol) {
      return false;
    }

    if (expected_.argmins.empty()) {
      return true;
    }

    double min_dx = std::numeric_limits<double>::infinity();
    for (double a : expected_.argmins) {
      const double dx = std::abs(o.argmin - a);
      min_dx = std::min(min_dx, dx);
    }

    const double dx_tol = 5.0 * input_.accuracy;

    if (expected_.argmins.size() == 1U) {
      return min_dx <= dx_tol;
    }

    return true;
  }

  InType GetTestInputData() final {
    return input_;
  }

 private:
  void PrepareTestCase(const FuncParam &param) {
    test_case_ = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(param);
    input_ = test_case_.problem;
    expected_ = test_case_.expected;
  }

  LocalTestType test_case_{};
  InType input_{};
  ExpectedSolution expected_{};
};

namespace {

std::vector<FuncParam> BuildFunctionalTestParams() {
  const auto tests = LoadTestCasesFromData();
  return BuildTestTasks(tests);
}

class FunctionalTestInstance : public SizovDGlobalSearchFunctionalTests {
 public:
  explicit FunctionalTestInstance(FuncParam param) : param_(std::move(param)) {}

  void TestBody() override {
    RunFunctionalTestCase(param_);
  }

 private:
  FuncParam param_;
};

struct FunctionalTestRegistrar {
  explicit FunctionalTestRegistrar(std::vector<FuncParam> tasks) : tasks_(std::move(tasks)) {
    test_names_.reserve(tasks_.size());
    std::size_t idx = 0;
    for (const auto &param : tasks_) {
      const LocalTestType &local = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(param);
      const std::string &test_name = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kNameTest)>(param);
      test_names_.push_back(test_name + "_" + local.name);
      const std::size_t current_idx = idx++;

      ::testing::RegisterTest(
          "GlobalSearchFunctionalTests", test_names_.back().c_str(), nullptr, nullptr, __FILE__, __LINE__,
          [this, current_idx]() -> ::testing::Test * { return new FunctionalTestInstance(tasks_[current_idx]); });
    }
  }

 private:
  std::vector<FuncParam> tasks_;
  std::vector<std::string> test_names_;
};

const bool kRegisteredFunctionalTests = []() {
  static const FunctionalTestRegistrar kRegistrar(BuildFunctionalTestParams());
  (void)kRegistrar;
  return true;
}();

}  // namespace

}  // namespace sizov_d_global_search
