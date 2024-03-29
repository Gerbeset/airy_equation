#pragma once
#include <array>
#include <deal.II/lac/vector.h>

#include <vector>

using namespace dealii;

template <int stages>
void pre_step(const std::array<double, stages> weights,
              const std::vector<BlockVector<double>> &parabolic_data,
              const double tau, BlockVector<double> &rhs_data);
