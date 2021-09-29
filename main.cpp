#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <errno.h>
#include <cstdint>
#include <cassert>
#include <mpi.h>

using std::isnan;
using std::isfinite;

double integrated_func(double x) {
  assert(!isnan(x));
  assert(isfinite(x));

  return 4.f / (1.f + x * x);
}

double integrate_trapeze(double from, double to, uint64_t samples, double (*f)(double)) {
  assert(!isnan(from));
  assert(!isnan(to));
  assert(isfinite(from));
  assert(isfinite(to));
  assert(samples > 0);
  assert(f != NULL);
  assert(to >= from);
  
  if(from == to) 
    return 0;

  double integral = 0;
  double sample_length = (to - from) / samples;
  for(uint64_t sample = 0; sample < samples; ++sample) {
    double x_1 = sample * sample_length + from;
    double x_2 = (sample + 1) * sample_length + from;
    
    integral += (x_2 - x_1) * (f(x_1) + f(x_2)) / 2.f;
  }

  return integral;
}

int main(int argc, char *argv[]) {
  if(argc < 2) return -EINVAL;
  const uint64_t n_samples = strtol(argv[1], NULL, 10);

  MPI_Init(&argc, &argv);

  int world_size = 0;
  int world_rank = 0;

  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  if (world_rank == 0) {
    printf("Main process: %lf\n", integrate_trapeze(0, 1, n_samples, integrated_func));
  }
  
  uint64_t samples_per_process = n_samples / world_size;
  if(world_rank == 0)
    /* 
    ** If sample number is not a multiple of world_size, the main process will
    ** take care of leftover samples
    */
    samples_per_process += n_samples % world_size;
  
  const double sample_size = 1.f / n_samples;

  double local_integral = 0;
  double from = world_rank * samples_per_process * sample_size;
  if (world_rank != 0)
    from += sample_size * (n_samples % world_size);

  double to = from + samples_per_process * sample_size;
  local_integral = integrate_trapeze(from, to, samples_per_process, integrated_func);
  printf("Subprocess [%d]: %lf on [%lf, %lf]\n", world_rank, local_integral, from, to);

  double integral_value = 0;

  MPI_Reduce(&local_integral, &integral_value, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  
  if(world_rank == 0) {
    printf("Main process parallel: %lf\n", integral_value);
  }

  MPI_Finalize();

  return 0;
}
