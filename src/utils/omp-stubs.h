#ifndef __OMP_STUBS_H__
#define __OMP_STUBS_H__

/*! \file omp-stubs.h
 * \brief implement stub OpenMP functions used for debugging without OpenMP
 */

const int omp_get_num_procs() { return 1; }
const int omp_get_thread_num() { return 0; }
const double omp_get_wtime() { return 0.0; }


#endif
