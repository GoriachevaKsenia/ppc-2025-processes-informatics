#include "dolov_v_torus_topology/seq/include/ops_seq.hpp"

#include <utility>
#include <vector>

#include "dolov_v_torus_topology/common/include/common.hpp"

namespace dolov_v_torus_topology {

DolovVTorusTopologySEQ::DolovVTorusTopologySEQ(InType in) : internal_input_(std::move(in)) {
  SetTypeOfTask(GetStaticTypeOfTask());
}

bool DolovVTorusTopologySEQ::ValidationImpl() {
  return true;
}

bool DolovVTorusTopologySEQ::PreProcessingImpl() {
  internal_output_.route.clear();
  internal_output_.received_message.clear();
  return true;
}

bool DolovVTorusTopologySEQ::RunImpl() {
  internal_output_.received_message = internal_input_.message;
  internal_output_.route = {internal_input_.sender_rank, internal_input_.receiver_rank};

  return true;
}

bool DolovVTorusTopologySEQ::PostProcessingImpl() {
  GetOutput() = internal_output_;
  return true;
}

}  // namespace dolov_v_torus_topology
