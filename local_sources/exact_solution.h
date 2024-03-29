#pragma once

#include <array>
#include <bits/types/time_t.h>
#include <deal.II/base/exceptions.h>
#include <deal.II/base/numbers.h>
#include <deal.II/base/function.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/base/convergence_table.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/numerics/vector_tools_common.h>
#include <deal.II/numerics/vector_tools_integrate_difference.h>
#include <deal.II/base/function_time.h>
#include <deal.II/numerics/vector_tools_interpolate.h>
#include <fstream>
#include <iostream>
#include <ostream>
#include <vector>
#include <cmath>

#include <deal.II/base/function.h>

using namespace dealii; 

template<int dim>
class ExactSolution : public Function<dim> 
{
  public: 
    ExactSolution(); 

    virtual double value(const Point<dim> &p, const unsigned int component) const override;
};
