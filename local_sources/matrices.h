#pragma once

#include <array>
#include <deal.II/base/exceptions.h>
#include <deal.II/base/numbers.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/tensor.h>
#include <deal.II/fe/fe_update_flags.h>
#include <deal.II/lac/block_sparse_matrix.h>
#include <deal.II/lac/block_sparsity_pattern.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/sparsity_pattern.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/grid/tria.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/numerics/vector_tools_common.h>
#include <fstream>
#include <iostream>
#include <ostream>
#include <vector>
#include <cmath>

using namespace dealii; 

template<int dim>
class matrices_for_problem
{
    public: 
    matrices_for_problem(const FE_Q<dim> &in_fe); 
    BlockSparseMatrix<double> system_matrix; 
    BlockSparseMatrix<double> rhs_mass; 
    BlockSparseMatrix<double> rhs_stiff; 
    BlockSparseMatrix<double> restriction_matrix; 
    BlockSparseMatrix<double> restriction_rhs; 
    BlockSparseMatrix<double> mass_update_matrix; 

    SparseDirectUMFPACK system_solver; 
    SparseDirectUMFPACK restriction_solver; 
    SparseDirectUMFPACK mass_solver; 
    
    void initialize(const SparsityPattern &in_sparsity_pattern, const AffineConstraints<double> &in_constraints); 

    void construct_matrices(const DoFHandler<dim> &dof_handler, 
    const double &tau, 
    const double &stab); 

    void construct_matrices_2D(const DoFHandler<dim> &dof_handler, 
    const double &tau, 
    const double &stab); 

    // Sets a new time step in the system matrix. Useful for when the problem requires a new timestep at every update
    void set_new_timestep(const DoFHandler<dim> &dof_handler, const double &tau, const double &stab); 

    void set_new_timestep_2D(const DoFHandler<dim> &dof_handler, const double &tau, const double &stab);

    private: 
    FE_Q<dim> fe; 
    BlockSparsityPattern block_sparsity_pattern;
    AffineConstraints<double> constraints; 
};