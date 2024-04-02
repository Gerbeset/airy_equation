#include "matrices.h"
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/lac/full_matrix.h>
#include <iostream>

template <int dim>
matrices_for_problem<dim>::matrices_for_problem(const FE_Q<dim> &in_fe)
    : fe(in_fe) {}

template <int dim>
void matrices_for_problem<dim>::initialize(
    const SparsityPattern &in_sparsity_pattern,
    const AffineConstraints<double> &in_constraints) {
  constraints.copy_from(in_constraints);

  block_sparsity_pattern.reinit(dim + 1, dim + 1);

  for (unsigned int i = 0; i < dim + 1; ++i) {
    for (unsigned int j = 0; j < dim + 1; ++j) {
      block_sparsity_pattern.block(i, j).copy_from(in_sparsity_pattern);
    }
  }

  block_sparsity_pattern.collect_sizes();

  system_matrix.reinit(block_sparsity_pattern);
  rhs_mass.reinit(block_sparsity_pattern);
  rhs_stiff.reinit(block_sparsity_pattern);
  restriction_matrix.reinit(block_sparsity_pattern);
  restriction_rhs.reinit(block_sparsity_pattern);
  mass_update_matrix.reinit(block_sparsity_pattern);
}

template <int dim>
void matrices_for_problem<dim>::construct_matrices(
    const DoFHandler<dim> &dof_handler, const double &tau, const double &stab) {
  // std::cout << "Constructing matrices with time step =" << tau << std::endl;
  QGauss<dim> quadrature_formula(fe.degree + 1);
  FEValues<dim> fe_values(fe, quadrature_formula,
                          update_values | update_gradients |
                              update_quadrature_points | update_JxW_values);

  const unsigned int dofs_per_cell = fe.n_dofs_per_cell();
  const unsigned int n_q_points = quadrature_formula.size();

  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

  if (dim == 1) {
    double m_ij = 0;
    double a_ij = 0;
    double n_x_ij = 0;
    double b_x_ij = 0;

    FullMatrix<double> local_rhs_mass_00(dofs_per_cell, dofs_per_cell);

    FullMatrix<double> local_system_00(dofs_per_cell, dofs_per_cell);
    FullMatrix<double> local_system_01(dofs_per_cell, dofs_per_cell);
    FullMatrix<double> local_system_10(dofs_per_cell, dofs_per_cell);
    FullMatrix<double> local_system_11(dofs_per_cell, dofs_per_cell);

    FullMatrix<double> local_rhs_stiff_00(dofs_per_cell, dofs_per_cell);
    FullMatrix<double> local_rhs_stiff_01(dofs_per_cell, dofs_per_cell);

    FullMatrix<double> local_restriction_rhs_10(dofs_per_cell, dofs_per_cell);

    FullMatrix<double> local_restricition_matrix_11(dofs_per_cell,
                                                    dofs_per_cell);

    FullMatrix<double> local_mass_update_matrix_00(dofs_per_cell,
                                                   dofs_per_cell);
    FullMatrix<double> local_mass_update_matrix_10(dofs_per_cell,
                                                   dofs_per_cell);
    FullMatrix<double> local_mass_update_matrix_11(dofs_per_cell,
                                                   dofs_per_cell);

    const FEValuesViews::Scalar<dim> phi(
        fe_values, 0); // phi corresponds to the u-variable, ie, first component

    for (const auto &cell :
         dof_handler.active_cell_iterators()) // Loop over cells
    {
      fe_values.reinit(cell); // Reset of the cell

      local_rhs_mass_00 = 0;

      local_system_00 = 0;
      local_system_01 = 0;
      local_system_10 = 0;
      local_system_11 = 0;

      local_rhs_stiff_00 = 0;
      local_rhs_stiff_01 = 0;

      local_restriction_rhs_10 = 0;

      local_restricition_matrix_11 = 0;

      local_mass_update_matrix_00 = 0;
      local_mass_update_matrix_10 = 0;
      local_mass_update_matrix_11 = 0;

      Tensor<1, dim> B;
      B[0] = 1;

      for (unsigned int q = 0; q < n_q_points; ++q) {
        for (unsigned int i = 0; i < dofs_per_cell; ++i) {
          const double phi_i = phi.value(i, q);
          const Tensor<1, dim> grad_phi_i = phi.gradient(i, q);

          for (unsigned int j = 0; j < dofs_per_cell;
               ++j) // Loop over local dofs again
          {
            const double phi_j = phi.value(j, q);

            const Tensor<1, dim> grad_phi_j = phi.gradient(j, q);

            // std::cout<<"New fe_values.JxW(q) =
            // "<<fe_values.JxW(q)<<std::endl;

            m_ij = (phi_i * phi_j) * fe_values.JxW(q);
            a_ij = (grad_phi_i * grad_phi_j) * fe_values.JxW(q);
            n_x_ij = (B * grad_phi_i * phi_j) * fe_values.JxW(q);
            b_x_ij = (phi_i * B * grad_phi_j) * fe_values.JxW(q);

            // std::cout<<"In constructor: m_ij = "<<m_ij<<std::endl;
            // std::cout<<"In constructor: a_ij = "<<a_ij<<std::endl;

            local_rhs_mass_00(i, j) += m_ij;

            local_system_00(i, j) += m_ij + stab * tau * a_ij;
            local_system_01(i, j) += -tau * a_ij - tau * stab * n_x_ij;
            local_system_10(i, j) += -stab * b_x_ij + a_ij;
            local_system_11(i, j) += stab * m_ij - n_x_ij;

            local_restriction_rhs_10(i, j) += stab * b_x_ij - a_ij;

            local_restricition_matrix_11(i, j) += stab * m_ij - n_x_ij;

            // Observe that this compute G(U), whereas the system matrix has
            // -G(U). This is so that in IMEX stepping we only have to use
            // tau>0, whereas before we were doing -tau.

            local_rhs_stiff_00(i, j) += -stab * a_ij;
            local_rhs_stiff_01(i, j) += a_ij + stab * n_x_ij;

            local_mass_update_matrix_00(i, j) += m_ij;
            local_mass_update_matrix_10(i, j) += a_ij - stab * b_x_ij;
            local_mass_update_matrix_11(i, j) += stab * m_ij - n_x_ij;
          }
        }
      }
      cell->get_dof_indices(local_dof_indices);
      constraints.distribute_local_to_global(
          local_rhs_mass_00, local_dof_indices, rhs_mass.block(0, 0));

      constraints.distribute_local_to_global(local_system_00, local_dof_indices,
                                             system_matrix.block(0, 0));
      constraints.distribute_local_to_global(local_system_01, local_dof_indices,
                                             system_matrix.block(0, 1));
      constraints.distribute_local_to_global(local_system_10, local_dof_indices,
                                             system_matrix.block(1, 0));
      constraints.distribute_local_to_global(local_system_11, local_dof_indices,
                                             system_matrix.block(1, 1));

      constraints.distribute_local_to_global(local_restriction_rhs_10,
                                             local_dof_indices,
                                             restriction_rhs.block(1, 0));

      constraints.distribute_local_to_global(local_restricition_matrix_11,
                                             local_dof_indices,
                                             restriction_matrix.block(1, 1));

      constraints.distribute_local_to_global(
          local_rhs_stiff_00, local_dof_indices, rhs_stiff.block(0, 0));
      constraints.distribute_local_to_global(
          local_rhs_stiff_01, local_dof_indices, rhs_stiff.block(0, 1));

      constraints.distribute_local_to_global(local_mass_update_matrix_00,
                                             local_dof_indices,
                                             mass_update_matrix.block(0, 0));
      constraints.distribute_local_to_global(local_mass_update_matrix_10,
                                             local_dof_indices,
                                             mass_update_matrix.block(1, 0));
      constraints.distribute_local_to_global(local_mass_update_matrix_11,
                                             local_dof_indices,
                                             mass_update_matrix.block(1, 1));
    }

    for (unsigned int j = 0; j < dof_handler.n_dofs(); ++j) {
      restriction_rhs.block(0, 0).diag_element(j) = 1;
      restriction_matrix.block(0, 0).diag_element(j) = 1;
    }

#if 0
    std::cout << "Constructed matrix = " << std::endl;
    system_matrix.print_formatted(std::cout);
    std::cout << std::endl;
#endif

    system_solver.initialize(system_matrix);
    restriction_solver.initialize(restriction_matrix);
    mass_solver.initialize(mass_update_matrix);
  }

  if (dim == 2) {
    double m_ij = 0;
    double a_ij = 0;
    double a_x_ij = 0;
    double a_x_y_ij = 0;
    double n_x_ij = 0;
    double n_y_ij = 0;
    double b_x_ij = 0;
    double b_y_ij = 0;

    FullMatrix<double> local_rhs_mass_00(dofs_per_cell, dofs_per_cell);

    FullMatrix<double> local_system_00(dofs_per_cell, dofs_per_cell);
    FullMatrix<double> local_system_01(dofs_per_cell, dofs_per_cell);
    FullMatrix<double> local_system_02(dofs_per_cell, dofs_per_cell);
    FullMatrix<double> local_system_10(dofs_per_cell, dofs_per_cell);
    FullMatrix<double> local_system_11(dofs_per_cell, dofs_per_cell);
    FullMatrix<double> local_system_20(dofs_per_cell, dofs_per_cell);
    FullMatrix<double> local_system_22(dofs_per_cell, dofs_per_cell);

    FullMatrix<double> local_rhs_stiff_00(dofs_per_cell, dofs_per_cell);
    FullMatrix<double> local_rhs_stiff_01(dofs_per_cell, dofs_per_cell);
    FullMatrix<double> local_rhs_stiff_02(dofs_per_cell, dofs_per_cell);

    FullMatrix<double> local_restriction_rhs_10(dofs_per_cell, dofs_per_cell);
    FullMatrix<double> local_restriction_rhs_20(dofs_per_cell, dofs_per_cell);

    FullMatrix<double> local_restriction_matrix_11(dofs_per_cell,
                                                   dofs_per_cell);
    FullMatrix<double> local_restriction_matrix_22(dofs_per_cell,
                                                   dofs_per_cell);

    FullMatrix<double> local_mass_update_matrix_00(dofs_per_cell,
                                                   dofs_per_cell);
    FullMatrix<double> local_mass_update_matrix_10(dofs_per_cell,
                                                   dofs_per_cell);
    FullMatrix<double> local_mass_update_matrix_11(dofs_per_cell,
                                                   dofs_per_cell);
    FullMatrix<double> local_mass_update_matrix_20(dofs_per_cell,
                                                   dofs_per_cell);
    FullMatrix<double> local_mass_update_matrix_22(dofs_per_cell,
                                                   dofs_per_cell);

    const FEValuesViews::Scalar<dim> phi(
        fe_values, 0); // phi corresponds to the u-variable, ie, first component

    for (const auto &cell :
         dof_handler.active_cell_iterators()) // Loop over cells
    {
      fe_values.reinit(cell); // Reset of the cell

      local_rhs_mass_00 = 0;

      local_system_00 = 0;
      local_system_01 = 0;
      local_system_02 = 0;
      local_system_02 = 0;
      local_system_10 = 0;
      local_system_11 = 0;
      local_system_20 = 0;
      local_system_22 = 0;

      local_rhs_stiff_00 = 0;
      local_rhs_stiff_01 = 0;
      local_rhs_stiff_02 = 0;

      local_restriction_rhs_10 = 0;
      local_restriction_rhs_20 = 0;

      local_restriction_matrix_11 = 0;
      local_restriction_matrix_22 = 0;

      local_mass_update_matrix_00 = 0;
      local_mass_update_matrix_10 = 0;
      local_mass_update_matrix_11 = 0;
      local_mass_update_matrix_20 = 0;
      local_mass_update_matrix_22 = 0;

      Tensor<1, dim> D_x;
      Tensor<1, dim> D_y;
      D_x[0] = D_y[1] = 1;
      D_x[1] = D_y[0] = 0;

      for (unsigned int q = 0; q < n_q_points; ++q) {
        for (unsigned int i = 0; i < dofs_per_cell; ++i) {
          const double phi_i = phi.value(i, q);
          const Tensor<1, dim> grad_phi_i = phi.gradient(i, q);

          for (unsigned int j = 0; j < dofs_per_cell;
               ++j) // Loop over local dofs again
          {
            const double phi_j = phi.value(j, q);
            const Tensor<1, dim> grad_phi_j = phi.gradient(j, q);

            m_ij = (phi_i * phi_j) * fe_values.JxW(q);
            a_ij = (grad_phi_i * grad_phi_j) * fe_values.JxW(q);
            a_x_ij = (D_x * grad_phi_i * D_x * grad_phi_j) * fe_values.JxW(q);
            a_x_y_ij = (D_y * grad_phi_i * D_x * grad_phi_j) * fe_values.JxW(q);
            n_x_ij = (D_x * grad_phi_i * phi_j) * fe_values.JxW(q);
            n_y_ij = (D_y * grad_phi_i * phi_j) * fe_values.JxW(q);
            b_x_ij = (phi_i * D_x * grad_phi_j) * fe_values.JxW(q);
            b_y_ij = (phi_i * D_y * grad_phi_j) * fe_values.JxW(q);

            // std::cout<<"New: m_"<<i<<","<<j<<" = "<< m_ij <<std::endl;

            local_rhs_mass_00(i, j) += m_ij;

            // std::cout<<"New: rhs_mass_cell("<<i<<","<<j<<") = "<<
            // local_rhs_mass_00(i,j) <<std::endl;

            local_system_00(i, j) += m_ij + stab * tau * a_ij;
            local_system_01(i, j) += -tau * a_x_ij - tau * stab * n_x_ij;
            local_system_02(i, j) += -tau * a_x_y_ij - tau * stab * n_y_ij;
            local_system_10(i, j) += -stab * b_x_ij + a_x_ij;
            local_system_11(i, j) += stab * m_ij - n_x_ij;
            local_system_20(i, j) += a_x_y_ij - stab * b_y_ij;
            local_system_22(i, j) += stab * m_ij - n_x_ij;

            local_restriction_rhs_10(i, j) += stab * b_x_ij - a_x_ij;
            local_restriction_rhs_20(i, j) += stab * b_y_ij - a_x_y_ij;

            local_restriction_matrix_11(i, j) += stab * m_ij - n_x_ij;
            local_restriction_matrix_22(i, j) += stab * m_ij - n_x_ij;

            local_rhs_stiff_00(i, j) += -stab * a_ij;
            local_rhs_stiff_01(i, j) += a_x_ij + stab * n_x_ij;
            local_rhs_stiff_02(i, j) += a_x_y_ij + stab * n_y_ij;

            local_mass_update_matrix_00(i, j) += m_ij;
            local_mass_update_matrix_10(i, j) += a_x_ij - stab * b_x_ij;
            local_mass_update_matrix_11(i, j) += stab * m_ij - n_x_ij;
            local_mass_update_matrix_20(i, j) += a_x_y_ij - stab * b_y_ij;
            local_mass_update_matrix_22(i, j) += stab * m_ij - n_x_ij;
          }
        }
      }
      cell->get_dof_indices(local_dof_indices);
      constraints.distribute_local_to_global(
          local_rhs_mass_00, local_dof_indices, rhs_mass.block(0, 0));

      constraints.distribute_local_to_global(local_system_00, local_dof_indices,
                                             system_matrix.block(0, 0));
      constraints.distribute_local_to_global(local_system_01, local_dof_indices,
                                             system_matrix.block(0, 1));
      constraints.distribute_local_to_global(local_system_02, local_dof_indices,
                                             system_matrix.block(0, 2));
      constraints.distribute_local_to_global(local_system_10, local_dof_indices,
                                             system_matrix.block(1, 0));
      constraints.distribute_local_to_global(local_system_11, local_dof_indices,
                                             system_matrix.block(1, 1));
      constraints.distribute_local_to_global(local_system_20, local_dof_indices,
                                             system_matrix.block(2, 0));
      constraints.distribute_local_to_global(local_system_22, local_dof_indices,
                                             system_matrix.block(2, 2));

      constraints.distribute_local_to_global(local_restriction_rhs_10,
                                             local_dof_indices,
                                             restriction_rhs.block(1, 0));
      constraints.distribute_local_to_global(local_restriction_rhs_20,
                                             local_dof_indices,
                                             restriction_rhs.block(2, 0));

      constraints.distribute_local_to_global(local_restriction_matrix_11,
                                             local_dof_indices,
                                             restriction_matrix.block(1, 1));
      constraints.distribute_local_to_global(local_restriction_matrix_22,
                                             local_dof_indices,
                                             restriction_matrix.block(2, 2));

      constraints.distribute_local_to_global(
          local_rhs_stiff_00, local_dof_indices, rhs_stiff.block(0, 0));
      constraints.distribute_local_to_global(
          local_rhs_stiff_01, local_dof_indices, rhs_stiff.block(0, 1));
      constraints.distribute_local_to_global(
          local_rhs_stiff_02, local_dof_indices, rhs_stiff.block(0, 2));

      constraints.distribute_local_to_global(local_mass_update_matrix_00,
                                             local_dof_indices,
                                             mass_update_matrix.block(0, 0));
      constraints.distribute_local_to_global(local_mass_update_matrix_10,
                                             local_dof_indices,
                                             mass_update_matrix.block(1, 0));
      constraints.distribute_local_to_global(local_mass_update_matrix_11,
                                             local_dof_indices,
                                             mass_update_matrix.block(1, 1));
      constraints.distribute_local_to_global(local_mass_update_matrix_20,
                                             local_dof_indices,
                                             mass_update_matrix.block(2, 0));
      constraints.distribute_local_to_global(local_restriction_matrix_22,
                                             local_dof_indices,
                                             restriction_matrix.block(2, 2));
    }

    for (unsigned int j = 0; j < dof_handler.n_dofs(); ++j) {
      restriction_rhs.block(0, 0).diag_element(j) = 1;
      restriction_matrix.block(0, 0).diag_element(j) = 1;
    }

    system_solver.initialize(system_matrix);
    // restriction_solver.initialize(restriction_matrix);
    // mass_solver.initialize(mass_update_matrix);
  }
}

template <int dim>
void matrices_for_problem<dim>::set_new_timestep(
    const DoFHandler<dim> &dof_handler, const double &tau, const double &stab) {
  //std::cout << "Reconstructing matrices with time step = " << tau << std::endl;

  for (unsigned int i = 0; i < dim + 1; ++i) {
    system_matrix.block(0, i) = 0;
  }

  QGauss<dim> quadrature_formula(fe.degree + 1);
  FEValues<dim> fe_values(fe, quadrature_formula,
                          update_values | update_gradients |
                              update_quadrature_points | update_JxW_values);

  const unsigned int dofs_per_cell = fe.n_dofs_per_cell();
  const unsigned int n_q_points = quadrature_formula.size();

  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

  if (dim == 1) {
    double m_ij = 0;
    double a_ij = 0;
    double n_x_ij = 0;

    FullMatrix<double> local_system_00(dofs_per_cell, dofs_per_cell);
    FullMatrix<double> local_system_01(dofs_per_cell, dofs_per_cell);
    FullMatrix<double> local_system_10(dofs_per_cell, dofs_per_cell);
    FullMatrix<double> local_system_11(dofs_per_cell, dofs_per_cell);

    const FEValuesViews::Scalar<dim> phi(
        fe_values, 0); // phi corresponds to the u-variable, ie, first component

    for (const auto &cell :
         dof_handler.active_cell_iterators()) // Loop over cells
    {
      fe_values.reinit(cell); // Reset of the cell

      local_system_00 = 0;
      local_system_01 = 0;
      local_system_10 = 0;
      local_system_11 = 0;

      Tensor<1, dim> B;
      B[0] = 1;

      for (unsigned int q = 0; q < n_q_points; ++q) {
        for (unsigned int i = 0; i < dofs_per_cell; ++i) {
          const double phi_i = phi.value(i, q);
          const Tensor<1, dim> grad_phi_i = phi.gradient(i, q);

          for (unsigned int j = 0; j < dofs_per_cell;
               ++j) // Loop over local dofs again
          {
            const double phi_j = phi.value(j, q);

            const Tensor<1, dim> grad_phi_j = phi.gradient(j, q);

            // std::cout<<"New fe_values.JxW(q) =
            // "<<fe_values.JxW(q)<<std::endl;

            m_ij = (phi_i * phi_j) * fe_values.JxW(q);
            a_ij = (grad_phi_i * grad_phi_j) * fe_values.JxW(q);
            n_x_ij = (B * grad_phi_i * phi_j) * fe_values.JxW(q);

            // std::cout<<"In constructor: m_ij = "<<m_ij<<std::endl;
            // std::cout<<"In constructor: a_ij = "<<a_ij<<std::endl;

            local_system_00(i, j) += m_ij + stab * tau * a_ij;
            local_system_01(i, j) += -tau * a_ij - tau * stab * n_x_ij;
          }
        }
      }
      cell->get_dof_indices(local_dof_indices);

      constraints.distribute_local_to_global(local_system_00, local_dof_indices,
                                             system_matrix.block(0, 0));
      constraints.distribute_local_to_global(local_system_01, local_dof_indices,
                                             system_matrix.block(0, 1));
    }
    #if 0
    std::cout << "Reconstructed matrix = " << std::endl;
    system_matrix.print_formatted(std::cout);
    std::cout << std::endl;
    #endif 
  }

  if (dim == 2) {
    double m_ij = 0;
    double a_ij = 0;
    double a_x_ij = 0;
    double a_x_y_ij = 0;
    double n_x_ij = 0;
    double n_y_ij = 0;
    double b_x_ij = 0;
    double b_y_ij = 0;

    FullMatrix<double> local_rhs_mass_00(dofs_per_cell, dofs_per_cell);

    FullMatrix<double> local_system_00(dofs_per_cell, dofs_per_cell);
    FullMatrix<double> local_system_01(dofs_per_cell, dofs_per_cell);
    FullMatrix<double> local_system_02(dofs_per_cell, dofs_per_cell);
    FullMatrix<double> local_system_10(dofs_per_cell, dofs_per_cell);
    FullMatrix<double> local_system_11(dofs_per_cell, dofs_per_cell);
    FullMatrix<double> local_system_20(dofs_per_cell, dofs_per_cell);
    FullMatrix<double> local_system_22(dofs_per_cell, dofs_per_cell);

    FullMatrix<double> local_rhs_stiff_00(dofs_per_cell, dofs_per_cell);
    FullMatrix<double> local_rhs_stiff_01(dofs_per_cell, dofs_per_cell);
    FullMatrix<double> local_rhs_stiff_02(dofs_per_cell, dofs_per_cell);

    FullMatrix<double> local_restriction_rhs_10(dofs_per_cell, dofs_per_cell);
    FullMatrix<double> local_restriction_rhs_20(dofs_per_cell, dofs_per_cell);

    FullMatrix<double> local_restriction_matrix_11(dofs_per_cell,
                                                   dofs_per_cell);
    FullMatrix<double> local_restriction_matrix_22(dofs_per_cell,
                                                   dofs_per_cell);

    FullMatrix<double> local_mass_update_matrix_00(dofs_per_cell,
                                                   dofs_per_cell);
    FullMatrix<double> local_mass_update_matrix_10(dofs_per_cell,
                                                   dofs_per_cell);
    FullMatrix<double> local_mass_update_matrix_11(dofs_per_cell,
                                                   dofs_per_cell);
    FullMatrix<double> local_mass_update_matrix_20(dofs_per_cell,
                                                   dofs_per_cell);
    FullMatrix<double> local_mass_update_matrix_22(dofs_per_cell,
                                                   dofs_per_cell);

    const FEValuesViews::Scalar<dim> phi(
        fe_values, 0); // phi corresponds to the u-variable, ie, first component

    for (const auto &cell :
         dof_handler.active_cell_iterators()) // Loop over cells
    {
      fe_values.reinit(cell); // Reset of the cell

      local_rhs_mass_00 = 0;

      local_system_00 = 0;
      local_system_01 = 0;
      local_system_02 = 0;
      local_system_02 = 0;
      local_system_10 = 0;
      local_system_11 = 0;
      local_system_20 = 0;
      local_system_22 = 0;

      local_rhs_stiff_00 = 0;
      local_rhs_stiff_01 = 0;
      local_rhs_stiff_02 = 0;

      local_restriction_rhs_10 = 0;
      local_restriction_rhs_20 = 0;

      local_restriction_matrix_11 = 0;
      local_restriction_matrix_22 = 0;

      local_mass_update_matrix_00 = 0;
      local_mass_update_matrix_10 = 0;
      local_mass_update_matrix_11 = 0;
      local_mass_update_matrix_20 = 0;
      local_mass_update_matrix_22 = 0;

      Tensor<1, dim> D_x;
      Tensor<1, dim> D_y;
      D_x[0] = D_y[1] = 1;
      D_x[1] = D_y[0] = 0;

      for (unsigned int q = 0; q < n_q_points; ++q) {
        for (unsigned int i = 0; i < dofs_per_cell; ++i) {
          const double phi_i = phi.value(i, q);
          const Tensor<1, dim> grad_phi_i = phi.gradient(i, q);

          for (unsigned int j = 0; j < dofs_per_cell;
               ++j) // Loop over local dofs again
          {
            const double phi_j = phi.value(j, q);
            const Tensor<1, dim> grad_phi_j = phi.gradient(j, q);

            m_ij = (phi_i * phi_j) * fe_values.JxW(q);
            a_ij = (grad_phi_i * grad_phi_j) * fe_values.JxW(q);
            a_x_ij = (D_x * grad_phi_i * D_x * grad_phi_j) * fe_values.JxW(q);
            a_x_y_ij = (D_y * grad_phi_i * D_x * grad_phi_j) * fe_values.JxW(q);
            n_x_ij = (D_x * grad_phi_i * phi_j) * fe_values.JxW(q);
            n_y_ij = (D_y * grad_phi_i * phi_j) * fe_values.JxW(q);
            b_x_ij = (phi_i * D_x * grad_phi_j) * fe_values.JxW(q);
            b_y_ij = (phi_i * D_y * grad_phi_j) * fe_values.JxW(q);

            // std::cout<<"New: m_"<<i<<","<<j<<" = "<< m_ij <<std::endl;

            // std::cout<<"New: rhs_mass_cell("<<i<<","<<j<<") = "<<
            // local_rhs_mass_00(i,j) <<std::endl;

            local_system_00(i, j) += m_ij + stab * tau * a_ij;
            local_system_01(i, j) += -tau * a_x_ij - tau * stab * n_x_ij;
            local_system_02(i, j) += -tau * a_x_y_ij - tau * stab * n_y_ij;
            local_system_10(i, j) += -stab * b_x_ij + a_x_ij;
            local_system_11(i, j) += stab * m_ij - n_x_ij;
            local_system_20(i, j) += a_x_y_ij - stab * b_y_ij;
            local_system_22(i, j) += stab * m_ij - n_x_ij;
          }
        }
      }
      cell->get_dof_indices(local_dof_indices);

      constraints.distribute_local_to_global(local_system_00, local_dof_indices,
                                             system_matrix.block(0, 0));
      constraints.distribute_local_to_global(local_system_01, local_dof_indices,
                                             system_matrix.block(0, 1));
      constraints.distribute_local_to_global(local_system_02, local_dof_indices,
                                             system_matrix.block(0, 2));
      constraints.distribute_local_to_global(local_system_10, local_dof_indices,
                                             system_matrix.block(1, 0));
      constraints.distribute_local_to_global(local_system_11, local_dof_indices,
                                             system_matrix.block(1, 1));
      constraints.distribute_local_to_global(local_system_20, local_dof_indices,
                                             system_matrix.block(2, 0));
      constraints.distribute_local_to_global(local_system_22, local_dof_indices,
                                             system_matrix.block(2, 2));
    }

    for (unsigned int j = 0; j < dof_handler.n_dofs(); ++j) {
      restriction_rhs.block(0, 0).diag_element(j) = 1;
      restriction_matrix.block(0, 0).diag_element(j) = 1;
    }

    system_solver.initialize(system_matrix);
    // restriction_solver.initialize(restriction_matrix);
    // mass_solver.initialize(mass_update_matrix);
  }
}

#if 0

template <int dim>
void matrices_for_problem<dim>::set_new_timestep(
    const DoFHandler<dim> &dof_handler, const double &tau, const double &stab) {
  for (unsigned int i = 0; i < dim + 1; ++i) {
    system_matrix.block(0, i) = 0.;
  }

  std::cout << "Check 1: " << std::endl;
  system_matrix.print_formatted(std::cout);
  std::cout << std::endl;

  std::cout << "Reconstructing matrices with new time step: " << tau
            << std::endl;

  QGauss<dim> quadrature_formula(fe.degree + 1);
  FEValues<dim> fe_values(fe, quadrature_formula,
                          update_values | update_gradients |
                              update_quadrature_points | update_JxW_values);

  const unsigned int dofs_per_cell = fe.n_dofs_per_cell();
  const unsigned int n_q_points = quadrature_formula.size();

  if (dim == 1) {
    double m_ij = 0;
    double a_ij = 0;
    double n_x_ij = 0;

    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

    FullMatrix<double> local_system_00(dofs_per_cell, dofs_per_cell);
    FullMatrix<double> local_system_01(dofs_per_cell, dofs_per_cell);

    const FEValuesViews::Scalar<dim> phi(
        fe_values, 0); // phi corresponds to the u-variable, ie, first component

    for (const auto &cell :
         dof_handler.active_cell_iterators()) // Loop over cells
    {
      fe_values.reinit(cell); // Reset of the cell

      local_system_00 = 0;
      local_system_01 = 0;

      Tensor<1, dim> B;
      B[0] = 1;

      for (unsigned int q = 0; q < n_q_points; ++q) {
        for (unsigned int i = 0; i < dofs_per_cell; ++i) {
          const double phi_i = phi.value(i, q);
          const Tensor<1, dim> grad_phi_i = phi.gradient(i, q);

          for (unsigned int j = 0; j < dofs_per_cell;
               ++j) // Loop over local dofs again
          {
            const double phi_j = phi.value(j, q);

            const Tensor<1, dim> grad_phi_j = phi.gradient(j, q);

            // std::cout<<"New fe_values.JxW(q) =
            // "<<fe_values.JxW(q)<<std::endl;

            m_ij = (phi_i * phi_j) * fe_values.JxW(q);
            a_ij = (grad_phi_i * grad_phi_j) * fe_values.JxW(q);

            // std::cout<<"In reconstructor: m_ij = "<<m_ij<<std::endl;
            // std::cout<<"In reconstructor: a_ij = "<<a_ij<<std::endl;
            local_system_00(i, j) += m_ij + stab * tau * a_ij;
            local_system_01(i, j) += -tau * a_ij - tau * stab * n_x_ij;


          }

          std::cout<<"In reconstructor: local_system_00 = "<<std::endl; 
          local_system_00.print(std::cout);
          std::cout<<"In reconstructor: local_system_01 = "<<std::endl; 
          local_system_01.print(std::cout); 
        }

        // local_rhs_mass_00.print_formatted(std::cout);
        // std::cout<<std::endl;
      }
      cell->get_dof_indices(local_dof_indices);
      constraints.distribute_local_to_global(local_system_00, local_dof_indices,
                                             system_matrix.block(0, 0));
      constraints.distribute_local_to_global(local_system_01, local_dof_indices,
                                             system_matrix.block(0, 1));
    }
  }

  if (dim == 2) {
    double m_ij = 0;
    double a_ij = 0;
    double a_x_ij = 0;
    double a_x_y_ij = 0;
    double n_x_ij = 0;
    double n_y_ij = 0;

    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

    FullMatrix<double> local_system_00(dofs_per_cell, dofs_per_cell);
    FullMatrix<double> local_system_01(dofs_per_cell, dofs_per_cell);
    FullMatrix<double> local_system_02(dofs_per_cell, dofs_per_cell);

    const FEValuesViews::Scalar<dim> phi(
        fe_values, 0); // phi corresponds to the u-variable, ie, first component

    for (const auto &cell :
         dof_handler.active_cell_iterators()) // Loop over cells
    {
      fe_values.reinit(cell); // Reset of the cell

      local_system_00 = 0;
      local_system_01 = 0;
      local_system_02 = 0;

      Tensor<1, dim> D_x;
      Tensor<1, dim> D_y;
      D_x[0] = D_y[1] = 1;
      D_x[1] = D_y[0] = 0;

      for (unsigned int q = 0; q < n_q_points; ++q) {
        for (unsigned int i = 0; i < dofs_per_cell; ++i) {
          const double phi_i = phi.value(i, q);
          const Tensor<1, dim> grad_phi_i = phi.gradient(i, q);

          for (unsigned int j = 0; j < dofs_per_cell;
               ++j) // Loop over local dofs again
          {
            const double phi_j = phi.value(j, q);

            const Tensor<1, dim> grad_phi_j = phi.gradient(j, q);

            m_ij = (phi_i * phi_j) * fe_values.JxW(q);
            a_ij = (grad_phi_i * grad_phi_j) * fe_values.JxW(q);
            a_x_ij = (D_x * grad_phi_i * D_x * grad_phi_j) * fe_values.JxW(q);
            a_x_y_ij = (D_y * grad_phi_i * D_x * grad_phi_j) * fe_values.JxW(q);
            n_x_ij = (D_x * grad_phi_i * phi_j) * fe_values.JxW(q);
            n_y_ij = (D_y * grad_phi_i * phi_j) * fe_values.JxW(q);

            local_system_00(i, j) += m_ij + stab * tau * a_ij;
            local_system_01(i, j) += -tau * a_x_ij - tau * stab * n_x_ij;
            local_system_02(i, j) += -tau * a_x_y_ij - tau * stab * n_y_ij;
          }
        }

        // local_rhs_mass_00.print_formatted(std::cout);
        // std::cout<<std::endl;
      }
      cell->get_dof_indices(local_dof_indices);
      constraints.distribute_local_to_global(local_system_00, local_dof_indices,
                                             system_matrix.block(0, 0));
      constraints.distribute_local_to_global(local_system_01, local_dof_indices,
                                             system_matrix.block(0, 1));
      constraints.distribute_local_to_global(local_system_02, local_dof_indices,
                                             system_matrix.block(0, 2));
    }
  }
  std::cout << "New constructed matrix:" << std::endl;
  system_matrix.print_formatted(std::cout);
}
#endif

#if 0
template <int dim>
void matrices_for_problem<dim>::set_new_timestep_2D(
    const DoFHandler<dim> &dof_handler, const double &tau, const double &stab) {
  for (unsigned int i = 0; i < dim + 1; ++i) {
    system_matrix.block(0, i) = 0.;
  }

  QGauss<dim> quadrature_formula(fe.degree + 1);
  FEValues<dim> fe_values(fe, quadrature_formula,
                          update_values | update_gradients |
                              update_quadrature_points | update_JxW_values);

  const unsigned int dofs_per_cell = fe.n_dofs_per_cell();
  const unsigned int n_q_points = quadrature_formula.size();
  double m_ij = 0;
  double a_ij = 0;
  double a_x_ij = 0;
  double a_x_y_ij = 0;
  double n_x_ij = 0;
  double n_y_ij = 0;

  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

  FullMatrix<double> local_system_00(dofs_per_cell, dofs_per_cell);
  FullMatrix<double> local_system_01(dofs_per_cell, dofs_per_cell);
  FullMatrix<double> local_system_02(dofs_per_cell, dofs_per_cell);

  const FEValuesViews::Scalar<dim> phi(
      fe_values, 0); // phi corresponds to the u-variable, ie, first component

  for (const auto &cell :
       dof_handler.active_cell_iterators()) // Loop over cells
  {
    fe_values.reinit(cell); // Reset of the cell

    local_system_00 = 0;
    local_system_01 = 0;
    local_system_02 = 0;

    Tensor<1, dim> D_x;
    Tensor<1, dim> D_y;
    D_x[0] = D_y[1] = 1;
    D_x[1] = D_y[0] = 0;

    for (unsigned int q = 0; q < n_q_points; ++q) {
      for (unsigned int i = 0; i < dofs_per_cell; ++i) {
        const double phi_i = phi.value(i, q);
        const Tensor<1, dim> grad_phi_i = phi.gradient(i, q);

        for (unsigned int j = 0; j < dofs_per_cell;
             ++j) // Loop over local dofs again
        {
          const double phi_j = phi.value(j, q);

          const Tensor<1, dim> grad_phi_j = phi.gradient(j, q);

          m_ij = (phi_i * phi_j) * fe_values.JxW(q);
          a_ij = (grad_phi_i * grad_phi_j) * fe_values.JxW(q);
          a_x_ij = (D_x * grad_phi_i * D_x * grad_phi_j) * fe_values.JxW(q);
          a_x_y_ij = (D_y * grad_phi_i * D_x * grad_phi_j) * fe_values.JxW(q);
          n_x_ij = (D_x * grad_phi_i * phi_j) * fe_values.JxW(q);
          n_y_ij = (D_y * grad_phi_i * phi_j) * fe_values.JxW(q);

          local_system_00(i, j) += m_ij + stab * tau * a_ij;
          local_system_01(i, j) += -tau * a_x_ij - tau * stab * n_x_ij;
          local_system_02(i, j) += -tau * a_x_y_ij - tau * stab * n_y_ij;
        }
      }
    }
    cell->get_dof_indices(local_dof_indices);
    constraints.distribute_local_to_global(local_system_00, local_dof_indices,
                                           system_matrix.block(0, 0));
    constraints.distribute_local_to_global(local_system_01, local_dof_indices,
                                           system_matrix.block(0, 1));
    constraints.distribute_local_to_global(local_system_02, local_dof_indices,
                                           system_matrix.block(0, 2));
  }
    }
#endif

#if 0
template <int dim>
void matrices_for_problem<dim>::construct_matrices_2D(
    const DoFHandler<dim> &dof_handler, const double &tau, const double &stab) {
  QGauss<dim> quadrature_formula(fe.degree + 1);
  FEValues<dim> fe_values(fe, quadrature_formula,
                          update_values | update_gradients |
                              update_quadrature_points | update_JxW_values);

  const unsigned int dofs_per_cell = fe.n_dofs_per_cell();
  const unsigned int n_q_points = quadrature_formula.size();
  double m_ij = 0;
  double a_ij = 0;
  double a_x_ij = 0;
  double a_x_y_ij = 0;
  double n_x_ij = 0;
  double n_y_ij = 0;
  double b_x_ij = 0;
  double b_y_ij = 0;

  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

  // Change this for higher dimensions; need to think of a way to make this
  // dimensionless... Can we make an array of FullMatrices?

  FullMatrix<double> local_rhs_mass_00(dofs_per_cell, dofs_per_cell);

  FullMatrix<double> local_system_00(dofs_per_cell, dofs_per_cell);
  FullMatrix<double> local_system_01(dofs_per_cell, dofs_per_cell);
  FullMatrix<double> local_system_02(dofs_per_cell, dofs_per_cell);
  FullMatrix<double> local_system_10(dofs_per_cell, dofs_per_cell);
  FullMatrix<double> local_system_11(dofs_per_cell, dofs_per_cell);
  FullMatrix<double> local_system_20(dofs_per_cell, dofs_per_cell);
  FullMatrix<double> local_system_22(dofs_per_cell, dofs_per_cell);

  FullMatrix<double> local_rhs_stiff_00(dofs_per_cell, dofs_per_cell);
  FullMatrix<double> local_rhs_stiff_01(dofs_per_cell, dofs_per_cell);
  FullMatrix<double> local_rhs_stiff_02(dofs_per_cell, dofs_per_cell);

  FullMatrix<double> local_restriction_rhs_10(dofs_per_cell, dofs_per_cell);
  FullMatrix<double> local_restriction_rhs_20(dofs_per_cell, dofs_per_cell);

  FullMatrix<double> local_restriction_matrix_11(dofs_per_cell, dofs_per_cell);
  FullMatrix<double> local_restriction_matrix_22(dofs_per_cell, dofs_per_cell);

  FullMatrix<double> local_mass_update_matrix_00(dofs_per_cell, dofs_per_cell);
  FullMatrix<double> local_mass_update_matrix_10(dofs_per_cell, dofs_per_cell);
  FullMatrix<double> local_mass_update_matrix_11(dofs_per_cell, dofs_per_cell);
  FullMatrix<double> local_mass_update_matrix_20(dofs_per_cell, dofs_per_cell);
  FullMatrix<double> local_mass_update_matrix_22(dofs_per_cell, dofs_per_cell);

  const FEValuesViews::Scalar<dim> phi(
      fe_values, 0); // phi corresponds to the u-variable, ie, first component

  for (const auto &cell :
       dof_handler.active_cell_iterators()) // Loop over cells
  {
    fe_values.reinit(cell); // Reset of the cell

    local_rhs_mass_00 = 0;

    local_system_00 = 0;
    local_system_01 = 0;
    local_system_02 = 0;
    local_system_02 = 0;
    local_system_10 = 0;
    local_system_11 = 0;
    local_system_20 = 0;
    local_system_22 = 0;

    local_rhs_stiff_00 = 0;
    local_rhs_stiff_01 = 0;
    local_rhs_stiff_02 = 0;

    local_restriction_rhs_10 = 0;
    local_restriction_rhs_20 = 0;

    local_restriction_matrix_11 = 0;
    local_restriction_matrix_22 = 0;

    local_mass_update_matrix_00 = 0;
    local_mass_update_matrix_10 = 0;
    local_mass_update_matrix_11 = 0;
    local_mass_update_matrix_20 = 0;
    local_mass_update_matrix_22 = 0;

    Tensor<1, dim> D_x;
    Tensor<1, dim> D_y;
    D_x[0] = D_y[1] = 1;
    D_x[1] = D_y[0] = 0;

    for (unsigned int q = 0; q < n_q_points; ++q) {
      for (unsigned int i = 0; i < dofs_per_cell; ++i) {
        const double phi_i = phi.value(i, q);
        const Tensor<1, dim> grad_phi_i = phi.gradient(i, q);

        for (unsigned int j = 0; j < dofs_per_cell;
             ++j) // Loop over local dofs again
        {
          const double phi_j = phi.value(j, q);
          const Tensor<1, dim> grad_phi_j = phi.gradient(j, q);

          // std::cout<<"New fe_values.JxW(q) = "<<fe_values.JxW(q)<<std::endl;

          m_ij = (phi_i * phi_j) * fe_values.JxW(q);
          a_ij = (grad_phi_i * grad_phi_j) * fe_values.JxW(q);
          a_x_ij = (D_x * grad_phi_i * D_x * grad_phi_j) * fe_values.JxW(q);
          a_x_y_ij = (D_y * grad_phi_i * D_x * grad_phi_j) * fe_values.JxW(q);
          n_x_ij = (D_x * grad_phi_i * phi_j) * fe_values.JxW(q);
          n_y_ij = (D_y * grad_phi_i * phi_j) * fe_values.JxW(q);
          b_x_ij = (phi_i * D_x * grad_phi_j) * fe_values.JxW(q);
          b_y_ij = (phi_i * D_y * grad_phi_j) * fe_values.JxW(q);

          // std::cout<<"New: m_"<<i<<","<<j<<" = "<< m_ij <<std::endl;

          local_rhs_mass_00(i, j) += m_ij;

          // std::cout<<"New: rhs_mass_cell("<<i<<","<<j<<") = "<<
          // local_rhs_mass_00(i,j) <<std::endl;

          local_system_00(i, j) += m_ij + stab * tau * a_ij;
          local_system_01(i, j) += -tau * a_x_ij - tau * stab * n_x_ij;
          local_system_02(i, j) += -tau * a_x_y_ij - tau * stab * n_y_ij;
          local_system_10(i, j) += -stab * b_x_ij + a_x_ij;
          local_system_11(i, j) += stab * m_ij - n_x_ij;
          local_system_20(i, j) += a_x_y_ij - stab * b_y_ij;
          local_system_22(i, j) += stab * m_ij - n_x_ij;

          local_restriction_rhs_10(i, j) += stab * b_x_ij - a_x_ij;
          local_restriction_rhs_20(i, j) += stab * b_y_ij - a_x_y_ij;

          local_restriction_matrix_11(i, j) += stab * m_ij - n_x_ij;
          local_restriction_matrix_22(i, j) += stab * m_ij - n_x_ij;

          local_rhs_stiff_00(i, j) += -stab * a_ij;
          local_rhs_stiff_01(i, j) += a_x_ij + stab * n_x_ij;
          local_rhs_stiff_02(i, j) += a_x_y_ij + stab * n_y_ij;

          local_mass_update_matrix_00(i, j) += m_ij;
          local_mass_update_matrix_10(i, j) += a_x_ij - stab * b_x_ij;
          local_mass_update_matrix_11(i, j) += stab * m_ij - n_x_ij;
          local_mass_update_matrix_20(i, j) += a_x_y_ij - stab * b_y_ij;
          local_mass_update_matrix_22(i, j) += stab * m_ij - n_x_ij;
        }
      }
    }
    cell->get_dof_indices(local_dof_indices);
    constraints.distribute_local_to_global(local_rhs_mass_00, local_dof_indices,
                                           rhs_mass.block(0, 0));

    constraints.distribute_local_to_global(local_system_00, local_dof_indices,
                                           system_matrix.block(0, 0));
    constraints.distribute_local_to_global(local_system_01, local_dof_indices,
                                           system_matrix.block(0, 1));
    constraints.distribute_local_to_global(local_system_02, local_dof_indices,
                                           system_matrix.block(0, 2));
    constraints.distribute_local_to_global(local_system_10, local_dof_indices,
                                           system_matrix.block(1, 0));
    constraints.distribute_local_to_global(local1.267e-02                                                                   
            1.267e-02 -2.533e-02  1.267e-02                                                        
                       1.267e-02 -2.533e-02  1.267e-02                                             
                                  1.267e-02 -2.533e-02  1.267e-02                                  
                                             1.267e-02 -2.533e-02  1.267e-02                       
                                                        1.267e-02 -2.533e-02  1.267e-02          _system_11, local_dof_indices,
                                           system_matrix.block(1, 1));
    constraints.distribute_local_to_global(local_system_20, local_dof_indices,
                                           system_matrix.block(2, 0));
    constraints.distribute_local_to_global(local_system_22, local_dof_indices,
                                           system_matrix.block(2, 2));

    constraints.distribute_local_to_global(local_restriction_rhs_10,
                                           local_dof_indices,
                                           restriction_rhs.block(1, 0));
    constraints.distribute_local_to_global(local_restriction_rhs_20,
                                           local_dof_indices,
                                           restriction_rhs.block(2, 0));

    constraints.distribute_local_to_global(local_restriction_matrix_11,
                                           local_dof_indices,
                                           restriction_matrix.block(1, 1));
    constraints.distribute_local_to_global(local_restriction_matrix_22,
                                           local_dof_indices,
                                           restriction_matrix.block(2, 2));

    constraints.distribute_local_to_global(
        local_rhs_stiff_00, local_dof_indices, rhs_stiff.block(0, 0));
    constraints.distribute_local_to_global(
        local_rhs_stiff_01, local_dof_indices, rhs_stiff.block(0, 1));
    constraints.distribute_local_to_global(
        local_rhs_stiff_02, local_dof_indices, rhs_stiff.block(0, 2));

    constraints.distribute_local_to_global(local_mass_update_matrix_00,
                                           local_dof_indices,
                                           mass_update_matrix.block(0, 0));
    constraints.distribute_local_to_global(local_mass_update_matrix_10,
                                           local_dof_indices,
                                           mass_update_matrix.block(1, 0));
    constraints.distribute_local_to_global(local_mass_update_matrix_11,
                                           local_dof_indices,
                                           mass_update_matrix.block(1, 1));
    constraints.distribute_local_to_global(local_mass_update_matrix_20,
                                           local_dof_indices,
                                           mass_update_matrix.block(2, 0));
    constraints.distribute_local_to_global(local_restriction_matrix_22,
                                           local_dof_indices,
                                           restriction_matrix.block(2, 2));
  }

  for (unsigned int j = 0; j < dof_handler.n_dofs(); ++j) {
    restriction_rhs.block(0, 0).diag_element(j) = 1;
    restriction_matrix.block(0, 0).diag_element(j) = 1;
  }

  system_solver.initialize(system_matrix);
}
#endif

template class matrices_for_problem<1>;
template class matrices_for_problem<2>;
template class matrices_for_problem<3>;
