#pragma once

#include "parabolic_module.h"
#include <deal.II/lac/block_vector.h>

template <int stages>
void pre_step(const std::array<double, stages> weights,
              const std::vector<BlockVector<double>> &parabolic_data,
              const double tau, BlockVector<double> &rhs_data) {

  rhs_data.block(0) = 0;
  rhs_data.block(1) = 0;

  for (unsigned int s = 0; s < stages; ++s) {
    // std::cout << "In airy: weights[" << s << "= " << weights[s] << std::endl;
    // std::cout << "In airy: parabolic_data[" << s
    //           << "] = " << parabolic_data[s].block(0) << std::endl
    //           << parabolic_data[s].block(1) << std::endl;

    rhs_data.add(tau * weights[s], parabolic_data[s]);
  }
  //std::cout << "rhs_data = " << rhs_data.block(0) << rhs_data.block(1)
  //          << std::endl
  //          << std::endl;
}

template void
pre_step<0>(const std::array<double, 0> weights,
            const std::vector<BlockVector<double>> &parabolic_data,
            const double tau, BlockVector<double> &rhs_data);

template void
pre_step<1>(const std::array<double, 1> weights,
            const std::vector<BlockVector<double>> &parabolic_data,
            const double tau, BlockVector<double> &rhs_data);

template void
pre_step<2>(const std::array<double, 2> weights,
            const std::vector<BlockVector<double>> &parabolic_data,
            const double tau, BlockVector<double> &rhs_data);
