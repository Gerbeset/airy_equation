#pragma once

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/lac/block_vector_base.h>
#include <deal.II/numerics/vector_tools_common.h>
#include <deal.II/lac/block_vector.h>
#include <fstream>
#include <iostream>
#include <ostream>
#include <vector>
#include <cmath>

using namespace dealii; 

class vectors_for_problem
{
    public: 
    vectors_for_problem();
    void initialize(const unsigned int &num_of_blocks, const unsigned int &components_per_block);

    BlockVector<double> solution; 
    BlockVector<double> old_solution; 
    BlockVector<double> data; 
    
    Vector<double> initial_condition; 
    Vector<double> final_solution; 

    #if 0
    Vector<double> solution; 
    Vector<double> old_solution; 
    Vector<double> rhs_data; 
    Vector<double> system_rhs; 
    Vector<double> initial_condition;
    Vector<double> exact_initial_condition; 
    Vector<double> wave_vector; 
    Vector<double> exact_solution_vector; 
    Vector<double> final_solution; 
    #endif
};



