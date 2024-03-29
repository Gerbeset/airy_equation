#pragma once
#include "compute_G.h"
#include "implicit_step.h"
#include "matrices.h"

using namespace dealii; 

class parabolic_solver {
public:
  parabolic_solver();

  template <int dim>
  void prepare(const matrices_for_problem<dim> &matrices,
               const AffineConstraints<double> &in_constraints);

  void parabolic_step(const BlockVector<double> &old_solution,
                      BlockVector<double> &solution,
                      const BlockVector<double> &rhs_data,
                      BlockVector<double> &new_parabolic_data);

  void set_new_solver(const BlockSparseMatrix<double> &new_matrix); 

private:
  implicit_step implict_step_;
  compute_G compute_g;
};
