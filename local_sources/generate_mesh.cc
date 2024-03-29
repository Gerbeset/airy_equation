#include "generate_mesh.h"

using namespace dealii; 

template<int dim> 
void generate_mesh(Triangulation<dim> &tria, const unsigned int number_of_refinements) 
{   

    if(dim == 1)
    {
    GridGenerator::hyper_cube(tria, 0, 1, true); // Mesh is on a square 
    tria.begin_active()->face(0)->set_boundary_id(0); // Label first boundary 
    tria.begin_active()->face(1)->set_boundary_id(1); // Label second boundary 
    tria.refine_global(number_of_refinements); // Refine
    }

    if(dim == 2)
    {
    GridGenerator::hyper_cube(tria, 0,1, true); // Mesh is on a square 
    tria.begin_active()->face(0)->set_boundary_id(0); // Label first boundary 
    tria.begin_active()->face(1)->set_boundary_id(2); // Label second boundary 
    tria.begin_active()->face(2)->set_boundary_id(1); // Label third boundary
    tria.begin_active()->face(3)->set_boundary_id(3); // Label fourth (and final) boundary 
    tria.refine_global(number_of_refinements);
    }
    
}


template void generate_mesh<1>(Triangulation<1> &tria, const unsigned int number_of_refinements); 
template void generate_mesh<2>(Triangulation<2> &tria, const unsigned int number_of_refinements); 
template void generate_mesh<3>(Triangulation<3> &tria, const unsigned int number_of_refinements); 
