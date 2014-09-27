#include "utils/omp-stubs.h"

const int omp_get_num_procs() { return 1; }
const int omp_get_thread_num() { return 0; }
const double omp_get_wtime() { return 0.0; }


