#pragma once
#include "parabolic_solver.h"
#include "matrices.h"
#include <deal.II/lac/block_matrix_base.h>

parabolic_solver::parabolic_solver() = default;

template <int dim>
void parabolic_solver::prepare(
    const matrices_for_problem<dim> &matrices,
    const AffineConstraints<double> &in_constraints) {
  implict_step_.initialize(matrices.system_matrix.get_sparsity_pattern(),
                           matrices.rhs_mass, matrices.restriction_rhs,
                           in_constraints);
  implict_step_.set_new_solver(matrices.system_matrix);
  implict_step_.set_new_restriction_solver(matrices.restriction_matrix);
  compute_g.initialize(matrices.rhs_stiff.get_sparsity_pattern(),
                       matrices.rhs_stiff, in_constraints);
}

void parabolic_solver::set_new_solver(
    const BlockSparseMatrix<double> &new_matrix) {
  implict_step_.set_new_solver(new_matrix);
}

void parabolic_solver::parabolic_step(const BlockVector<double> &old_solution,
                                      BlockVector<double> &solution,
                                      const BlockVector<double> &new_rhs_data,
                                      BlockVector<double> &new_parabolic_data) {
  implict_step_.step(old_solution, solution, new_rhs_data);
  compute_g.get_G_of_U(solution, new_parabolic_data);
}

template void parabolic_solver::prepare<1>(const matrices_for_problem<1> &matrices, const AffineConstraints<double> &in_constraints);
template void parabolic_solver::prepare<2>(const matrices_for_problem<2> &matrices, const AffineConstraints<double> &in_constraints);
