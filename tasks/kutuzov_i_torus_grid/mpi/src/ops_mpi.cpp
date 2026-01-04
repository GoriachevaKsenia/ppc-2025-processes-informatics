#include "kutuzov_i_torus_grid/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <cmath>
#include <string>
#include <tuple>
#include <vector>

#include "kutuzov_i_torus_grid/common/include/common.hpp"

namespace kutuzov_i_torus_grid {

KutuzovIThorusGridMPI::KutuzovIThorusGridMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = std::make_tuple(std::vector<int>{}, "");
}

bool KutuzovIThorusGridMPI::ValidationImpl() {
  int start = std::get<0>(GetInput());
  int end = std::get<1>(GetInput());

  if (start < 0 || end < 0) {
    return false;
  }

  int process_count = -1;
  MPI_Comm_size(MPI_COMM_WORLD, &process_count);

  if (start > process_count - 1) {
    return false;
  }
  if (end > process_count - 1) {
    return false;
  }

  return true;
}

bool KutuzovIThorusGridMPI::PreProcessingImpl() {
  return true;
}

bool KutuzovIThorusGridMPI::RunImpl() {
  int start = std::get<0>(GetInput());
  int end = std::get<1>(GetInput());

  int rank = -1;
  int process_count = -1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &process_count);

  std::string message;
  int message_size = -1;

  if (rank == start) {
    message = std::get<2>(GetInput());
    message_size = static_cast<int>(message.size());
  }

  if (start == end) {
    MPI_Bcast(&message_size, 1, MPI_INT, end, MPI_COMM_WORLD);
    message.resize(message_size);
    MPI_Bcast(message.data(), message_size, MPI_CHAR, end, MPI_COMM_WORLD);
    GetOutput() = std::make_tuple(std::vector<int>{}, message);
    return true;
  }

  if (process_count == 1) {
    GetOutput() = std::make_tuple(std::vector<int>{}, message);
    return true;
  }

  int columns = -1;
  int rows = -1;
  GenerateTopology(columns, rows, process_count);

  int sender = start;
  int receiver = -1;
  std::vector<int> route;
  int route_size = -1;

  int current_x = rank % columns;
  int current_y = rank / columns;
  int end_x = end % columns;
  int end_y = end / columns;

  while (sender != end) {
    if (rank == sender) {
      receiver = GetNextStep(current_x, current_y, end_x, end_y, columns, rows);
    }
    MPI_Bcast(&receiver, 1, MPI_INT, sender, MPI_COMM_WORLD);
    if (rank == sender) {
      MPI_Send(&message_size, 1, MPI_INT, receiver, 0, MPI_COMM_WORLD);
      MPI_Send(message.data(), message_size, MPI_CHAR, receiver, 1, MPI_COMM_WORLD);
      route.push_back(rank);
      route_size = static_cast<int>(route.size());
      MPI_Send(&route_size, 1, MPI_INT, receiver, 2, MPI_COMM_WORLD);
      MPI_Send(route.data(), route_size, MPI_INT, receiver, 3, MPI_COMM_WORLD);

    } else if (rank == receiver) {
      MPI_Recv(&message_size, 1, MPI_INT, sender, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      message.resize(message_size);
      MPI_Recv(message.data(), message_size, MPI_CHAR, sender, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      MPI_Recv(&route_size, 1, MPI_INT, sender, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      route.resize(route_size);
      MPI_Recv(route.data(), route_size, MPI_INT, sender, 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    sender = receiver;
  }

  if (rank == end) {
    route.push_back(rank);
    route_size = static_cast<int>(route.size());
  }

  MPI_Bcast(&route_size, 1, MPI_INT, end, MPI_COMM_WORLD);
  route.resize(route_size);
  MPI_Bcast(route.data(), route_size, MPI_INT, end, MPI_COMM_WORLD);
  MPI_Bcast(&message_size, 1, MPI_INT, end, MPI_COMM_WORLD);
  message.resize(message_size);
  MPI_Bcast(message.data(), message_size, MPI_CHAR, end, MPI_COMM_WORLD);

  GetOutput() = std::make_tuple(route, message);
  return true;
}

bool KutuzovIThorusGridMPI::PostProcessingImpl() {
  return true;
}

void KutuzovIThorusGridMPI::GenerateTopology(int &columns, int &rows, int process_count) {
  for (rows = static_cast<int>(std::sqrt(static_cast<double>(process_count))); rows > 0; rows--) {
    if (process_count % rows == 0) {
      columns = process_count / rows;
      break;
    }
  }
}

int KutuzovIThorusGridMPI::GetNextStep(int current_x, int current_y, int dest_x, int dest_y, int columns, int rows) {
  int up = (((current_y - 1 + rows) % rows) * columns) + current_x;
  int down = (((current_y + 1) % rows) * columns) + current_x;
  int left = (current_y * columns) + ((current_x - 1 + columns) % columns);
  int right = (current_y * columns) + ((current_x + 1) % columns);

  int mx = dest_x - current_x;
  int my = dest_y - current_y;

  if (abs(mx) > columns / 2) {
    mx = mx > 0 ? mx - columns : mx + columns;
  }
  if (abs(my) > rows / 2) {
    my = my > 0 ? my - rows : my + rows;
  }

  if (abs(mx) > abs(my)) {
    return mx >= 0 ? right : left;
  }

  return my >= 0 ? down : up;
}

}  // namespace kutuzov_i_torus_grid
