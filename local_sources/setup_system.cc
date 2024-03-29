#include "setup_system.h"
#include <deal.II/lac/affine_constraints.h>

template<int dim>
void setup_system(DoFHandler<dim> &in_dof_handler, 
FE_Q<dim> &in_fe, 
AffineConstraints<double> &in_constraints,
SparsityPattern &in_pattern)
{
    in_dof_handler.distribute_dofs(in_fe);          // Note here that we are going to have dofs = 2 x nodes since we have two components to our solution
  
    DoFRenumbering::component_wise(in_dof_handler);

    in_constraints.clear(); 
    
        for (int i = 0; i < dim; ++i)
    {                                   
    DoFTools::make_periodicity_constraints(in_dof_handler,           
                                          /*b_id1    */ i,
                                          /*b_id2    */ dim + i,
                                          /*direction*/ i,
                                          in_constraints); 
    }
     
    DoFTools::make_hanging_node_constraints(in_dof_handler, in_constraints); // Apply contstraints to dof_handler
    in_constraints.close();

    DynamicSparsityPattern dsp(in_dof_handler.n_dofs(), in_dof_handler.n_dofs()); //Standard sparsity matrix routine
    DoFTools::make_sparsity_pattern(in_dof_handler,
                                    dsp,
                                    in_constraints,
                                    /* keep_constrained_dofs = */ false);
    in_pattern.copy_from(dsp);
}


template void setup_system<1>(DoFHandler<1> &in_dof_handler, FE_Q<1> &in_fe, AffineConstraints<double> &in_constraints, SparsityPattern &in_pattern);
template void setup_system<2>(DoFHandler<2> &in_dof_handler, FE_Q<2> &in_fe, AffineConstraints<double> &in_constraints, SparsityPattern &in_pattern);
template void setup_system<3>(DoFHandler<3> &in_dof_handler, FE_Q<3> &in_fe, AffineConstraints<double> &in_constraints, SparsityPattern &in_pattern);