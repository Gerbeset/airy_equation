#pragma once  // Use include guards or #pragma once to prevent multiple inclusions

#include "compute_G.h"
using namespace dealii;

compute_G::compute_G()
{}

void compute_G::initialize(
    const BlockSparsityPattern &in_block_pattern, 
    const BlockSparseMatrix<double> &in_matrix, 
    const AffineConstraints<double> &in_constraints
    )
{
    g_matrix.reinit(in_block_pattern);
    if(g_matrix.n() != in_matrix.n())
    {
        std::cout<<"Error in compute_g.cc: length of in_matrix does not match length of g_matrix. Check to ensure that in_matrix is reinit using in_pattern"<<std::endl; 
        assert(false); 
    }

   if(g_matrix.m() != in_matrix.m())
    {
        std::cout<<"Error in compute_g.cc: width of in_matrix does not match width of g_matrix. Check to ensure that in_matrix is reinit using in_pattern"<<std::endl; 
        assert(false); 
    }

    g_matrix.copy_from(in_matrix); 
    constraints.copy_from(in_constraints); 
    temp.reinit(in_matrix.n_block_cols(), in_matrix.block(0,0).n()); 
}


void compute_G::get_G_of_U(const BlockVector<double> &U, BlockVector<double> &G_of_U)
{
    g_matrix.vmult(G_of_U, U); 
    constraints.set_zero(G_of_U.block(0));
    constraints.set_zero(G_of_U.block(1)); 
};