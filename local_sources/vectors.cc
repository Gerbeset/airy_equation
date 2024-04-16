#include "vectors.h"

vectors_for_problem::vectors_for_problem() {}

void vectors_for_problem::initialize(const unsigned int &num_of_blocks,
                                     const unsigned int &components_per_block) {
  solution.reinit(num_of_blocks, components_per_block);
  old_solution.reinit(num_of_blocks, components_per_block);
  data.reinit(num_of_blocks, components_per_block);
  restriction_data.reinit(num_of_blocks, components_per_block);
  restriction_solution.reinit(num_of_blocks, components_per_block); 

  initial_condition.reinit(components_per_block);
  final_solution.reinit(components_per_block);

#if 0 
    solution.reinit(N); 
    old_solution.reinit(N); 
    rhs_data.reinit(N); 
    system_rhs.reinit(N); 
    initial_condition.reinit(N);
    exact_initial_condition.reinit(N); 
    wave_vector.reinit(N); 
    exact_solution_vector.reinit(N); 
    final_solution.reinit(N);
#endif
}