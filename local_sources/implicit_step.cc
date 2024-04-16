#include "implicit_step.h"
#include <array>
#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/vector.h>
#include <iostream>

implicit_step::implicit_step() = default;

void implicit_step::initialize(
    const BlockSparsityPattern &in_block_pattern,
    const BlockSparseMatrix<double> &in_mass_matrix,
    const BlockSparseMatrix<double> &in_restriction_rhs_matrix,
    const AffineConstraints<double> &in_constraints) {

  rhs_mass.reinit(in_block_pattern);
  rhs_mass.copy_from(in_mass_matrix);

  restriction_rhs_matrix.reinit(in_block_pattern);
  restriction_rhs_matrix.copy_from(in_restriction_rhs_matrix);

  constraints.copy_from(in_constraints);
  U_new.reinit(in_mass_matrix.n());

  rhs_data.reinit(in_mass_matrix.n_block_cols(),
                  in_mass_matrix.block(0, 0).n());
  data.reinit(in_mass_matrix.n_block_cols(),
                  in_mass_matrix.block(0, 0).n()); 
}

void implicit_step::set_new_solver(
    const BlockSparseMatrix<double> &in_system_matrix) {
  system_solver.initialize(in_system_matrix);
}

void implicit_step::set_new_correction_solver(
    const BlockSparseMatrix<double> &in_correction_matrix) {
  correction_solver.initialize(in_correction_matrix);
};

void implicit_step::set_new_restriction_solver(
    const BlockSparseMatrix<double> &in_restriction_matrix) {
  restirction_solver.initialize(in_restriction_matrix);
};

void implicit_step::step(const BlockVector<double> &old_U,
                         BlockVector<double> &new_U,
                         const BlockVector<double> &new_rhs_data) {

  std::cout<<"new_rhs_data = "<<new_rhs_data.block(0)<<" "<<new_rhs_data.block(1)<<std::endl<<std::endl; 

  rhs_mass.vmult(rhs_data, old_U);
  constraints.set_zero(rhs_data.block(0));
  constraints.set_zero(rhs_data.block(1));

  //std::cout << "In airy: rhs_data at mass: " << rhs_data.block(0) 
  //          << rhs_data.block(1) << std::endl;

  rhs_data.add(1., new_rhs_data);
  constraints.set_zero(rhs_data.block(0));
  constraints.set_zero(rhs_data.block(1));

  //std::cout << "In airy: rhs_data after addition = " << rhs_data.block(0)
  //          << std::endl
  //          << rhs_data.block(1) << std::endl;

  system_solver.vmult(new_U, rhs_data);
  constraints.distribute(new_U.block(0));
  constraints.distribute(new_U.block(1));

  //std::cout << "In airy:: solution = " << new_U.block(0) << new_U.block(1)
  //          << std::endl
  //          << std::endl;
};

#if 0
void implicit_step::correction_step(const BlockVector<double> &old_U,
                     BlockVector<double> &new_U,
                     BlockVector<double> &new_rhs_data)
{
  rhs_mass.vmult(rhs_data, old_U);
  constraints.set_zero(rhs_data.block(0));
  constraints.set_zero(rhs_data.block(1));

  rhs_data.add(1., new_rhs_data);
  constraints.set_zero(rhs_data.block(0));
  constraints.set_zero(rhs_data.block(1));

  correction_solver.vmult(new_U, rhs_data);
  constraints.distribute(new_U.block(0));
  constraints.distribute(new_U.block(1));
}
#endif

#if 0
template <int stages>
void implicit_step::stage_step(
    const BlockVector<double> &U_old,
    BlockVector<double> &U_new,
    const std::array<double, stages> &weights,
    const std::vector<BlockVector<double>> &pre_computed_values,
    const double time_step)
{
  if (weights.size() != pre_computed_values.size()) {
    std::cout
        << "Error: number of coefficients and number of stages does not match"
        << std::endl;
    assert(false);
  }

  rhs_mass.vmult(rhs_data, U_old);
da
  constraints.set_zero(rhs_data.block(0));
  constraints.set_zero(rhs_data.block(1));

  for (unsigned int s = 0; s < stages; ++s) {
    rhs_data.add(time_step * weights[s], pre_computed_values[s]);
  }

  constraints.set_zero(rhs_data.block(0));
  constraints.set_zero(rhs_data.block(1));

  system_solver.vmult(U_new, rhs_data);
  constraints.distribute(U_new.block(0));
  constraints.distribute(U_new.block(1));
}
#endif

#if 0 
template <int stages>
void implicit_step::stage_correction_step(
    const BlockVector<double> &U_old,
    BlockVector<double> &U_new,
    const std::array<double, stages> &weights,
    const std::vector<BlockVector<double>> &pre_computed_values,
    const double time_step)
{
  if (weights.size() != pre_computed_values.size()) {
    std::cout
        << "Error: number of coefficients and number of stages does not match"
        << std::endl;
    assert(false);
  }

  rhs_mass.vmult(rhs_data, U_old);

  constraints.set_zero(rhs_data.block(0));
  constraints.set_zero(rhs_data.block(1));

  for (unsigned int s = 0; s < stages; ++s) {
    rhs_data.add(time_step * weights[s], pre_computed_values[s]);
  }

  constraints.set_zero(rhs_data.block(0));
  constraints.set_zero(rhs_data.block(1));

  correction_solver.vmult(U_new, rhs_data);
  constraints.distribute(U_new.block(0));
  constraints.distribute(U_new.block(1));
}
#endif

BlockVector<double>
implicit_step::backward_euler(const BlockVector<double> &U_old) {
  rhs_mass.vmult(rhs_data, U_old);
  constraints.set_zero(rhs_data.block(0));
  constraints.set_zero(rhs_data.block(1));

  system_solver.vmult(U_new, rhs_data);
  constraints.distribute(U_new.block(0));
  constraints.distribute(U_new.block(1));

  return U_new;
}

BlockVector<double>
implicit_step::crank_nicholson_step(const BlockVector<double> &U_old) {
  rhs_mass.vmult(rhs_data, U_old);
  constraints.distribute(rhs_data.block(0));
  constraints.distribute(rhs_data.block(1));

  // std::cout<<"In implicit_step.cc: rhs_data =
  // "<<rhs_data.block(0)<<rhs_data.block(1)<<std::endl<<std::endl;

  system_solver.vmult(U_new, rhs_data);
  constraints.distribute(U_new.block(0));
  constraints.distribute(U_new.block(1));

  U_new *= 2.0;
  U_new.add(-1., U_old);

  return U_new;
}

void implicit_step::solve_restriction(BlockVector<double> &U_old,
                                      BlockVector<double> &result) {

  //std::cout<<"restriction_rhs_matrix at solve"<<std::endl;             
  //restriction_rhs_matrix.print_formatted(std::cout);
  //std::cout<<std::endl; 

  restriction_rhs_matrix.vmult(data, U_old);
  constraints.set_zero(data.block(0));
  constraints.set_zero(data.block(1));

  // std::cout<<"restriction_rhs = "<<data.block(0)<<" "<<data.block(1)<<std::endl<<std::endl; 
  
  restirction_solver.vmult(result, data);
  constraints.distribute(result.block(0));
  constraints.distribute(result.block(1));  
  
  // std::cout<<"U_0 Z_0 = "<<result.block(0)<<" "<<result.block(1)<<std::endl<<std::endl; 
  
}

#if 0 
template void
implicit_step::stage_step<1>(const BlockVector<double> &,
                             BlockVector<double> &,
                             const std::array<double, 1> &,
                             const std::vector<BlockVector<double>> &,
                             const double);

template void
implicit_step::stage_step<2>(const BlockVector<double> &,
                             BlockVector<double> &,
                             const std::array<double, 2> &,
                             const std::vector<BlockVector<double>> &,
                             const double);

template void
implicit_step::stage_step<3>(const BlockVector<double> &,
                             BlockVector<double> &,
                             const std::array<double, 3> &,
                             const std::vector<BlockVector<double>> &,
                             const double);

template void
implicit_step::stage_step<4>(const BlockVector<double> &,
                             BlockVector<double> &,
                             const std::array<double, 4> &,
                             const std::vector<BlockVector<double>> &,
                             const double);

template void
implicit_step::stage_step<5>(const BlockVector<double> &,
                             BlockVector<double> &,
                             const std::array<double, 5> &,
                             const std::vector<BlockVector<double>> &,
                             const double);

template void
implicit_step::stage_step<6>(const BlockVector<double> &,
                             BlockVector<double> &,
                             const std::array<double, 6> &,
                             const std::vector<BlockVector<double>> &,
                             const double);

template void implicit_step::stage_correction_step<1>(
    const BlockVector<double> &,
    BlockVector<double> &,
    const std::array<double, 1> &,
    const std::vector<BlockVector<double>> &,
    const double);

template void implicit_step::stage_correction_step<2>(
    const BlockVector<double> &,
    BlockVector<double> &,
    const std::array<double, 2> &,
    const std::vector<BlockVector<double>> &,
    const double);

template void implicit_step::stage_correction_step<3>(
    const BlockVector<double> &,
    BlockVector<double> &,
    const std::array<double, 3> &,
    const std::vector<BlockVector<double>> &,
    const double);

template void implicit_step::stage_correction_step<4>(
    const BlockVector<double> &,
    BlockVector<double> &,
    const std::array<double, 4> &,
    const std::vector<BlockVector<double>> &,
    const double);

template void implicit_step::stage_correction_step<5>(
    const BlockVector<double> &,
    BlockVector<double> &,
    const std::array<double, 5> &,
    const std::vector<BlockVector<double>> &,
    const double);

template void implicit_step::stage_correction_step<6>(
    const BlockVector<double> &,
    BlockVector<double> &,
    const std::array<double, 6> &,
    const std::vector<BlockVector<double>> &,
    const double);
#endif
