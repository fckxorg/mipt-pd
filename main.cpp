#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <errno.h>
#include <cstdint>
#include <cassert>
#include <mpi.h>
#include <vector>

using std::isnan;
using std::isfinite;

class Timer {
  private:
    double t1;
    double t2;
    bool running;
  public:
    Timer() : t1(0), t2(0), running(false) {}
    void start() {
      assert(!running);
      t1 = MPI_Wtime();
      running = true;
    }

    void stop() {
      assert(running);
      t2 = MPI_Wtime();
      running = false;
    }

    double elapsed() {
      assert(!running);
      return t2 - t1;
    }
};

#ifdef VERBOSE 
void log(const char* format, ...) {
  va_list ptr;
  va_start(ptr, format);
  vprintf(format, ptr);
  va_end(ptr);
}
#else 
void log(...) {
 /* Empty body. */ 
}
#endif

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
  assert(f != NULL);
  assert(to >= from);
  
  if(from == to) 
    return 0;

  if(samples == 0)
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

#ifdef BENCHMARK
  Timer timer{};
#endif

  int world_size = 0;
  int world_rank = 0;

  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  if (world_rank == 0) {
#ifdef BENCHMARK
    timer.start();
#endif
    double single_process_integral = integrate_trapeze(0, 1, n_samples, integrated_func);
    log("Main process: %lf\n", single_process_integral);
#ifdef BENCHMARK
    timer.stop();
    printf("%lf\n", timer.elapsed());
#endif
  }
 
#ifdef BENCHMARK
  if(world_rank == 0)
    timer.start();
#endif
  double local_integral = 0;

  const double sample_size = 1.f / n_samples;
  uint64_t samples_per_process = ceil(static_cast<double>(n_samples) / world_size);
  double local_from = 0;
  double local_to = 0;
  
  std::vector<double> from = {};
  std::vector<double> to = {};
  
  if (world_rank == 0) {
    from.assign(world_size, 0);
    to.assign(world_size, 0);
    
    for(int i = 0; i < world_size; ++i) {
      from[i] = i * samples_per_process * sample_size;
      if(i == world_size - 1) {
        int local_samples = n_samples - samples_per_process * (world_size - 1); 
        to[i] = from[i] + local_samples * sample_size;
      }
      else {
        to[i] = from[i] + samples_per_process * sample_size;
      }
    }
  }

  MPI_Scatter(from.data(), 1, MPI_DOUBLE, &local_from, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Scatter(to.data(), 1, MPI_DOUBLE, &local_to, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  local_integral = integrate_trapeze(local_from, local_to, samples_per_process, integrated_func);
  log("Subprocess [%d]: %lf on [%lf, %lf]\n", world_rank, local_integral, local_from, local_to);

  double integral_value = 0;

  MPI_Reduce(&local_integral, &integral_value, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  
  if(world_rank == 0) {
    log("Main process parallel: %lf\n", integral_value);
  }

#ifdef BENCHMARK
  if(world_rank == 0) {
    timer.stop();
    printf("%lf\n", timer.elapsed());
  }
#endif
  
  MPI_Finalize();

  return 0;
}
