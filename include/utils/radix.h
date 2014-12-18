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
#ifndef QUICKRANK_UTILS_RADIX_H_
#define QUICKRANK_UTILS_RADIX_H_

#include <memory>

/*! \file radix.hpp
 * \brief Set of functions implementing descending radix sort for floating point values (ideal for long array)
 */

std::unique_ptr<unsigned int[]> idx_radixsort(float const* fvalues,
                                              const unsigned int nvalues);

//
// Functions belows are not used
//

/*! sort an array of float values without modifing the input array and returning permuted indexes of the sorted items
 *  @param fvalues input float array
 *  @param nvalues length of \a fvalues
 *  @return indexes of ascending sorted \a fvalues
 */
unsigned int *idxfloat_radixsort(float const* fvalues,
                                 const unsigned int nvalues);

enum sortorder {
  ascending,
  descending
};

/*! sort an array of float values
 *  @param fvalues input float array
 *  @param nvalues length of \a fvalues
 */
template<sortorder const order> void float_radixsort(
    float *fvalues, const unsigned int nvalues);

/*! sort an array of float values with respect to another one without modifing the input array and returning permuted indexes of the sorted items
 *  @param extvalues input float array
 *  @param fvalues input float array
 *  @param nvalues length of \a fvalues
 *  @return a sorted copy of \a extvalues wrt \a fvalues
 */
template<sortorder const order> float *copyextfloat_radixsort(
    float const* extvalues, float const* fvalues, const unsigned int nvalues);

#endif

