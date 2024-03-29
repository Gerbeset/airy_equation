#include "error_analysis.h"
#include "exact_solution.h"

using namespace dealii; 

template<int dim> 
void compute_error(const Triangulation<dim> &tria,
const Vector<double> &result, 
const double &time, 
const DoFHandler<dim> &in_dof_handler,
const FE_Q<dim> &in_fe)
{
    ExactSolution<dim> exact_solution; 
    exact_solution.set_time(time);

    Vector<float> difference_per_cell(tria.n_active_cells());
    Vector<float> exact_cell_L1_norm(tria.n_active_cells()); 
    Vector<float> exact_cell_Linf_norm(tria.n_active_cells()); 

    Vector<double> exact_solution_vector(in_dof_handler.n_dofs()); 
    Vector<double> zero_vector(in_dof_handler.n_dofs()); 

    VectorTools::interpolate(in_dof_handler, exact_solution, exact_solution_vector);

    ConvergenceTable convergence_table; 
    VectorTools::integrate_difference(in_dof_handler, result, exact_solution, difference_per_cell,
        QGauss<dim>(in_fe.degree+1), VectorTools::L1_norm); 
    const double pure_L1_error = VectorTools::compute_global_error(tria, difference_per_cell, VectorTools::L1_norm); 

    VectorTools::integrate_difference(in_dof_handler, zero_vector, exact_solution, exact_cell_L1_norm, 
            QGauss<dim>(in_fe.degree+1), VectorTools::L1_norm); 
    const double exact_L1_norm = VectorTools::compute_global_error(tria, exact_cell_L1_norm, VectorTools::L1_norm);
    const double relative_L1_error = pure_L1_error/exact_L1_norm; 

    VectorTools::integrate_difference(in_dof_handler, result, exact_solution, difference_per_cell, 
        QGauss<dim>(in_fe.degree+1), VectorTools::Linfty_norm); 
    const double pure_Linf_error = VectorTools::compute_global_error(tria, difference_per_cell, VectorTools::Linfty_norm); 

    VectorTools::integrate_difference(in_dof_handler, zero_vector, exact_solution, exact_cell_Linf_norm, 
        QGauss<dim>(in_fe.degree+1), VectorTools::Linfty_norm); 
    const double exact_Linf_norm = VectorTools::compute_global_error(tria, exact_cell_Linf_norm, VectorTools::Linfty_norm); 
    const double relative_Linf_error = pure_Linf_error/exact_Linf_norm; 


    convergence_table.add_value("L1", relative_L1_error); 
    convergence_table.add_value("Linf", relative_Linf_error); 

    convergence_table.set_precision("L1", 2); 
    convergence_table.set_precision("Linf", 2); 

    convergence_table.set_scientific("L1",true);
    convergence_table.set_scientific("Linf", true); 
    
    convergence_table.set_tex_caption("L1", "@f$L^1@f$ - error");
    convergence_table.set_tex_caption("Linf", "@f$ Linf@f$-error"); 

    convergence_table.write_text(std::cout); 
}

template void compute_error<1>(const Triangulation<1> &tria,
const Vector<double> &result, 
const double &time, 
const DoFHandler<1> &in_dof_handler,
const FE_Q<1> &in_fe);

template void compute_error<2>(const Triangulation<2> &tria,
const Vector<double> &result, 
const double &time, 
const DoFHandler<2> &in_dof_handler,
const FE_Q<2> &in_fe);

template void compute_error<3>(const Triangulation<3> &tria,
const Vector<double> &result, 
const double &time, 
const DoFHandler<3> &in_dof_handler,
const FE_Q<3> &in_fe);