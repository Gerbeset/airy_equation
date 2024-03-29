#include "compute_time_step.h"
using namespace dealii;

compute_time_step::compute_time_step(const double &in_mesh_size, 
const double &in_wave_speed, 
const double &in_final_time, const double &CFL)
{
  time_step = CFL*(in_mesh_size/in_wave_speed);
  double temp_val = in_final_time/time_step; 
  number_of_time_steps = static_cast<int>(temp_val); 
  time_step = in_final_time/number_of_time_steps; 
}
