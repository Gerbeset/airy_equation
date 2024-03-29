#include "interpolate_exact_at_time.h"
#include <deal.II/numerics/vector_tools_interpolate.h>

template<int dim>
void interpolate_exact_at_time(const DoFHandler<dim> &dof_handler, 
Vector<double> &data_vector, 
const double &time)
{
    ExactSolution<dim> exact_solution; 
    exact_solution.set_time(time); 

    VectorTools::interpolate(dof_handler, exact_solution, data_vector); 
}

template void interpolate_exact_at_time<1>(const DoFHandler<1> &dof_handler, Vector<double> &data_vector, const double &time);
template void interpolate_exact_at_time<2>(const DoFHandler<2> &dof_handler, Vector<double> &data_vector, const double &time);
template void interpolate_exact_at_time<3>(const DoFHandler<3> &dof_handler, Vector<double> &data_vector, const double &time);