#pragma once

#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/block_sparse_matrix.h>
#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/sparsity_pattern.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/affine_constraints.h>

using namespace dealii; 

class compute_G
{
    public :
    compute_G(); 
    void initialize(
        const BlockSparsityPattern &in_block_pattern, 
        const BlockSparseMatrix<double> &in_matrix, 
        const AffineConstraints<double> &in_constraints
    );

    void get_G_of_U(const BlockVector<double> &U, BlockVector<double> &G_of_U); 

    private :
    BlockSparseMatrix<double> g_matrix; 
    AffineConstraints<double> constraints; 
    BlockVector<double> temp; 
};