#pragma once

#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>
#include <iostream>
#include <fstream>
#include <cmath>

using namespace dealii; 

template<int dim> 
double compute_mesh_size(const Triangulation<dim> &tria);

//template<int dim> 
//double compute_mesh_size(const Triangulation<dim> &tria); 

#if 0
template<int dim> 
class compute_mesh_size
{
    public: 
    compute_mesh_size(); 
    double compute_h(const Triangulation<dim> &tria);
    double mesh_size; 

    private: 
    double calculate_mesh_size(Triangulation<dim> &tria); 
};
#endif 
