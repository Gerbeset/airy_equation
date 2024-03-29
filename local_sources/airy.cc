#include "compute_mesh_size.h"
#include "compute_time_step.h"
#include "error_analysis.h"
#include "generate_mesh.h"
#include "implicit_step.h"
#include "interpolate_exact_at_time.h"
#include "matrices.h"
#include "parabolic_module.h"
#include "parabolic_solver.h"
#include "setup_system.h"
#include "vectors.h"
#include <array>
#include <cmath>
#include <deal.II/base/exceptions.h>
#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/patterns.h>
#include <deal.II/base/timer.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/grid/tria.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/block_sparse_matrix.h>
#include <deal.II/lac/block_sparsity_pattern.h>
#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/sparse_matrix.h>
#include <fstream>
#include <iostream>
#include <numeric>
#include <ostream>
#include <string>
#include <vector>

/*

Changes to be made:

*/

using namespace dealii;

int main() {
  // Space dimension of the problem

  const unsigned int dim = 1;
  // Parameter handling
  ParameterHandler prm;
  prm.declare_entry("number_of_refinements", "1", Patterns::Integer(0));
  prm.declare_entry("poly_degree", "1", Patterns::Integer(1));
  prm.declare_entry("CFL", "1.0", Patterns::Double(0.00000000001));

  prm.parse_input("parameter.prm", "", true);

  const unsigned int refinement = prm.get_integer("number_of_refinements");
  const unsigned int poly_degree = prm.get_integer("poly_degree");
  const double CFL = prm.get_double("CFL");
  // End of parameters

  // Internal objects

  Timer timer;

  FE_Q<dim> fe(1);

  Triangulation<dim> triangulation;

  DoFHandler<dim> dof_handler(triangulation);
  AffineConstraints<double> constraints;
  SparsityPattern sparsity_pattern;

  double stab = 1;
  const unsigned int stages = 2;
  // const unsigned int stages = Butcher<Method::irk_2_2_1>::stages;
  const double final_time = 1 / (std::pow(2 * numbers::PI, 2));
  const double wave_speed = std::pow(2 * numbers::PI, 2);
  // End of internal objects

  generate_mesh(triangulation, refinement);

  setup_system(dof_handler, fe, constraints, sparsity_pattern);

  std::cout << "Number of dofs:" << dof_handler.n_dofs() << std::endl;

  double mesh_size = compute_mesh_size(triangulation);

  std::cout << "Mesh_size:" << mesh_size << std::endl;
  // std::cout<<"New: wave_speed:"<<wave_speed<<std::endl;

  compute_time_step time_step_data(mesh_size, wave_speed, final_time, CFL);

  double time_step = time_step_data.time_step;

  std::cout << "time_step =" << time_step << std::endl;

  const unsigned int number_of_time_steps = time_step_data.number_of_time_steps;

  std::cout << "number_of_time_steps:" << number_of_time_steps << std::endl;

  timer.start();

  matrices_for_problem<dim> matrices(fe);

  matrices.initialize(sparsity_pattern, constraints);
  matrices.construct_matrices(dof_handler, time_step, stab);

  matrices.set_new_timestep(dof_handler, time_step, stab);

  vectors_for_problem vectors;
  vectors.initialize(matrices.system_matrix.n_block_cols(),
                     matrices.system_matrix.block(0, 0).n());

  interpolate_exact_at_time(dof_handler, vectors.initial_condition, 0.0);

  double time = 0.0;
  unsigned int time_step_number = 0;

  vectors.old_solution.block(0) = vectors.initial_condition;

  // std::cout<<"Initial condition =
  // "<<vectors.old_solution.block(0)<<std::endl;

  std::vector<BlockVector<double>> parabolic_precomputed; // 2 = stages
  std::vector<BlockVector<double>> stage_U;

  BlockVector<double> temp;
  temp.reinit(matrices.system_matrix.n_block_cols(),
              matrices.system_matrix.block(0, 0).n());

  for (unsigned int i = 0; i < 3; ++i) {
    stage_U.push_back(temp);
  }

  for (unsigned int i = 0; i < 2; ++i) {
    parabolic_precomputed.push_back(temp);
  }

  parabolic_solver parabolic_solver_;
  parabolic_solver_.prepare<1>(matrices, constraints);

  implicit_step implicit;
  implicit.initialize(matrices.system_matrix.get_sparsity_pattern(),
                      matrices.rhs_mass, matrices.restriction_matrix,
                      constraints);
  implicit.set_new_solver(matrices.system_matrix);

  // std::cout << "Test 2" << std::endl;

  // std::array<double, 2> weights_1 = {};
  // std::array<double, 2> weights_2 = {1.};

  // DEBUGGING HERE
  std::cout << std::endl;
  while (time < final_time) {

    stage_U[0] = vectors.old_solution;

    // Stage 1
    // std::cout << "Airy: stage 1" << std::endl;
    matrices.set_new_timestep(dof_handler, time_step, stab);
    parabolic_solver_.set_new_solver(matrices.system_matrix);

    pre_step<0>({}, {}, time_step, vectors.data);
    parabolic_solver_.parabolic_step(stage_U[0], stage_U[1], vectors.data,
                                     parabolic_precomputed[1]);

    // std::cout << "In airy: stage_U[1] = " << stage_U[1].block(0)
    //           << stage_U[1].block(1) << std::endl;

    // Final stage
    //std::cout << "Airy: stage 2" << std::endl;
    matrices.set_new_timestep(dof_handler, 0, stab);
    parabolic_solver_.set_new_solver(matrices.system_matrix);

    pre_step<1>({1.}, {{parabolic_precomputed[1]}}, time_step, vectors.data);
    parabolic_solver_.parabolic_step(stage_U[1], stage_U[2], vectors.data,
                                     parabolic_precomputed[0]);

    //std::cout << "In airy: stage_U[2] = " << stage_U[2].block(0)
    //          << stage_U[2].block(1) << std::endl;

    vectors.old_solution = stage_U[2];

    time += 2*time_step;
    ++time_step_number;
    std::cout << std::endl << std::endl;
  }

// Second order method but inefficient

#if 0 
while (time_step_number < number_of_time_steps) {

    stage_U[0] = vectors.old_solution;

    // Stage 1
    // std::cout << "Airy: stage 1" << std::endl;
    matrices.set_new_timestep(dof_handler, 0.5 * time_step, stab);
    parabolic_solver_.set_new_solver(matrices.system_matrix);

    pre_step<0>({}, {}, time_step, vectors.data);
    parabolic_solver_.parabolic_step(stage_U[0], stage_U[1], vectors.data,
                                     parabolic_precomputed[1]);

    // std::cout << "In airy: stage_U[1] = " << stage_U[1].block(0)
    //           << stage_U[1].block(1) << std::endl;

    // Final stage
    //std::cout << "Airy: stage 2" << std::endl;
    matrices.set_new_timestep(dof_handler, 0, stab);
    parabolic_solver_.set_new_solver(matrices.system_matrix);

    pre_step<1>({0.5}, {{parabolic_precomputed[1]}}, time_step, vectors.data);
    parabolic_solver_.parabolic_step(stage_U[1], stage_U[2], vectors.data,
                                     parabolic_precomputed[0]);

    //std::cout << "In airy: stage_U[2] = " << stage_U[2].block(0)
    //          << stage_U[2].block(1) << std::endl;

    vectors.old_solution = stage_U[2];

    time += time_step;
    ++time_step_number;
    std::cout << std::endl << std::endl;
  }
#endif 


#if 0 
  while (time < final_time) {
  
    pre_step<0>({}, {}, time_step, vectors.rhs_data);
    parabolic_solver_.parabolic_step(vectors.old_solution, vectors.solution,
                                     vectors.rhs_data,
                                     parabolic_precomputed[1]);
    vectors.old_solution = vectors.solution; 
    time += time_step;

  }
#endif

#if 0
  while (time_step_number < number_of_time_steps) {
    ++time_step_number;
    time += time_step;

    pre_step<0>({}, {}, time_step, vectors.rhs_data);
    parabolic_solver_.parabolic_step(vectors.old_solution, vectors.solution,
                                     vectors.rhs_data,
                                     parabolic_precomputed[1]);
    vectors.old_solution = vectors.solution; 
  }
#endif

// LOW ORDER
#if 0 
  while(time_step_number<number_of_time_steps)
  {
    ++time_step_number; 
    time += time_step; 

    vectors.solution = implicit.backward_euler(vectors.old_solution); 
    vectors.old_solution = vectors.solution; 
  }
#endif

#if 0
  while (time_step_number < number_of_time_steps) {
    ++time_step_number;

    stage_U[0] = vectors.old_solution;

    double weight_1 = -std::accumulate(weights_1.begin(), weights_1.end(), -1.);

    matrices.set_new_timestep(dof_handler, weight_1 * time_step, stab);

    parabolic_solver_.set_new_solver(matrices.system_matrix);

    pre_step<2>(weights_1, parabolic_precomputed, time_step, vectors.rhs_data);

    parabolic_solver_.parabolic_step(stage_U[0], stage_U[1], vectors.rhs_data,
                                     parabolic_precomputed[1]);

    double weight_2 = -std::accumulate(weights_2.begin(), weights_2.end(), -1.);

    matrices.set_new_timestep(dof_handler, weight_2 * time_step, stab);

    parabolic_solver_.set_new_solver(matrices.system_matrix);

    pre_step<2>(weights_2, parabolic_precomputed, time_step, vectors.rhs_data);

    parabolic_solver_.parabolic_step(stage_U[1], stage_U[2], vectors.rhs_data,
                                     parabolic_precomputed[0]);
    vectors.solution = stage_U[2];

    vectors.old_solution = vectors.solution;

    time += stages * time_step;
  }
#endif

// LOW ORDER METHOD
#if 0
  while(time_step_number< number_of_time_steps)
  {
    ++time_step_number;  
    time += time_step; 

    matrices.rhs_mass.vmult( vectors.rhs_data, vectors.old_solution);
    constraints.set_zero(vectors.rhs_data); 

    //std::cout<<"New: rhs_data = "<< vectors.rhs_data.block(0) <<std::endl; 

    matrices.system_solver.vmult(vectors.solution, vectors.rhs_data); 
    constraints.distribute(vectors.solution); 

    vectors.old_solution = vectors.solution;
  }
#endif

  // vectors.final_solution = vectors.old_solution.block(0);
  // std::cout<<"New: final_solution = "<<vectors.final_solution<<std::endl;

  // compute_error(triangulation, vectors.final_solution, time, dof_handler,
  // fe);

  //  END LOW ORDER METHOD

#if 0 
while(time_step_number< number_of_time_steps)
{
  if(time_step_number % 25 ==0)
  {
    std::cout<<"Current time step = "<<time_step_number<<std::endl; 
  }
  time += time_step; 
  ++time_step_number; 

  vectors.solution = implicit.backward_euler(vectors.old_solution); 

  vectors.old_solution = vectors.solution;
}
#endif

#if 0
while(time_step_number< number_of_time_steps)
{
  time += time_step; 
  ++time_step_number; 

  vectors.solution = implicit.crank_nicholson_step(vectors.old_solution); 

  vectors.old_solution = vectors.solution;
}
#endif

#if 0 

/// High-order //

irk_stepping<dim, stages> irk(matrices, constraints); 

irk.solve_restriction(vectors.old_solution); 

while(time_step_number <number_of_time_steps)
    {
        if(time_step_number % 100 == 0.0)
        {
          std::cout<<"Current time step: "<< time_step_number<<std::endl; 
        }

      time += time_step; 
      ++time_step_number; 

      vectors.solution = irk.step(vectors.old_solution, Butcher<Method::irk_2_2_1>::tableau, time_step); 

      vectors.old_solution = vectors.solution; 
    }
#endif

#if 0 
  timer.stop();
  const double cpu_time = timer.wall_time();
  std::cout << "Time: " << cpu_time << " seconds " << std::endl;
#endif

  // End of time loop

  vectors.final_solution = vectors.old_solution.block(0);
  // std::cout<<"New: final_solution = "<<vectors.final_solution<<std::endl;

  // std::cout<<"New: time = "<<time<<std::endl;

  // std::cout << vectors.final_solution << std::endl;

  compute_error(triangulation, vectors.final_solution, time, dof_handler, fe);
}