#pragma once // Use include guards or #pragma once to prevent multiple
             // inclusions

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/block_sparse_matrix.h>
#include <deal.II/lac/block_sparsity_pattern.h>
#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/sparsity_pattern.h>
#include <deal.II/lac/vector.h>
#include <deal.II/numerics/data_out.h>
#include <fstream>
#include <iostream>

using namespace dealii;

class implicit_step
{
public:
  implicit_step();

  void initialize(const BlockSparsityPattern &block_pattern,
                  const BlockSparseMatrix<double> &in_mass_matrix,
                  const BlockSparseMatrix<double> &in_restriction_rhs_matrix,
                  const AffineConstraints<double> &in_constraints);

  void set_new_solver(const BlockSparseMatrix<double> &in_system_matrix);

  void set_new_correction_solver(
      const BlockSparseMatrix<double> &in_correction_matrix);

  void set_new_restriction_solver(
      const BlockSparseMatrix<double> &in_restriction_matrix);

  void step(const BlockVector<double> &old_U,
            BlockVector<double> &new_U,
            const BlockVector<double> &new_rhs_data);

  void correction_step(const BlockVector<double> &old_U,
                       BlockVector<double> &new_U,
                       BlockVector<double> &new_rhs_data);

  template <int stages>
  void stage_step(const BlockVector<double> &U_old,
                  BlockVector<double> &U_new,
                  const std::array<double, stages> &weights = {0.},
                  const std::vector<BlockVector<double>> &pre_computed_values =
                      std::vector<BlockVector<double>>(),
                  const double time_step = 0);

  template <int stages>
  void stage_correction_step(
      const BlockVector<double> &U_old,
      BlockVector<double> &U_new,
      const std::array<double, stages> &weights = {0.},
      const std::vector<BlockVector<double>> &pre_computed_values =
          std::vector<BlockVector<double>>(),
      const double time_step = 0);


  BlockVector<double> backward_euler(const BlockVector<double> &U_old);

  BlockVector<double> crank_nicholson_step(const BlockVector<double> &U_old);

  void solve_restriction(BlockVector<double> &U_old,
                         BlockVector<double> &result);

private:
  SparseDirectUMFPACK system_solver;
  SparseDirectUMFPACK correction_solver;
  SparseDirectUMFPACK restirction_solver;

  BlockSparseMatrix<double> rhs_mass;
  BlockSparseMatrix<double> restriction_rhs_matrix;

  BlockVector<double> U_new;
  BlockVector<double> rhs_data;
  BlockVector<double> data;

  BlockSparsityPattern sparsity_pattern;
  AffineConstraints<double> constraints;
};
