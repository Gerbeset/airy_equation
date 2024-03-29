#include "exact_solution.h"
#include <deal.II/base/numbers.h>
#include <cmath>


template <int dim>
ExactSolution<dim>::ExactSolution()
{}

template <int dim>
double ExactSolution<dim>::value(const Point<dim> &p, const unsigned int /*component*/) const
{
    const double time = this->get_time();
    double result; 
    result = cos(2 * numbers::PI * p[0] + std::pow(2 * numbers::PI, 3) * time);
    return result; 
}

// Explicit instantiation for the desired dimensions
template class ExactSolution<1>; 
template class ExactSolution<2>;
template class ExactSolution<3>;