#include "compute_mesh_size.h"

using namespace dealii; 

template<int dim>
double compute_mesh_size(const Triangulation<dim> &tria)
{
  double mesh_size = 0.0; 
  for (const auto &cell: tria.active_cell_iterators())
    {
      mesh_size = std::max(mesh_size, cell->diameter());
    }
  return mesh_size; 
}

template double compute_mesh_size<1>(const Triangulation<1> &tria);
template double compute_mesh_size<2>(const Triangulation<2> &tria);
template double compute_mesh_size<3>(const Triangulation<3> &tria); 
