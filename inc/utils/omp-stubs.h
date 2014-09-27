#ifndef QUICKRANK_UTILS_OMP_STUBS_H_
#define QUICKRANK_UTILS_OMP_STUBS_H_

/*! \file omp-stubs.h
 * \brief implement stub OpenMP functions used for debugging without OpenMP
 */

const int omp_get_num_procs();
const int omp_get_thread_num();
const double omp_get_wtime();


#endif
