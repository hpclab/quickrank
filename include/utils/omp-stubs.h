/*
 * QuickRank - A C++ suite of Learning to Rank algorithms
 * Webpage: http://quickrank.isti.cnr.it/
 * Contact: quickrank@isti.cnr.it
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 *
 * Contributor:
 *   HPC. Laboratory - ISTI - CNR - http://hpc.isti.cnr.it/
 */
#ifndef QUICKRANK_UTILS_OMP_STUBS_H_
#define QUICKRANK_UTILS_OMP_STUBS_H_

/*! \file omp-stubs.h
 * \brief implement stub OpenMP functions used for debugging without OpenMP
 */

const int omp_get_num_procs();
const int omp_get_thread_num();
const double omp_get_wtime();

#endif
