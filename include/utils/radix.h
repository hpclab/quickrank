/*
 * QuickRank - A C++ suite of Learning to Rank algorithms
 * Webpage: http://quickrank.isti.cnr.it/
 * Contact: quickrank@isti.cnr.it
 *
 * Unless explicitly acquired and licensed from Licensor under another
 * license, the contents of this file are subject to the Reciprocal Public
 * License ("RPL") Version 1.5, or subsequent versions as allowed by the RPL,
 * and You may not copy or use this file in either source code or executable
 * form, except in compliance with the terms and conditions of the RPL.
 *
 * All software distributed under the RPL is provided strictly on an "AS
 * IS" basis, WITHOUT WARRANTY OF ANY KIND, EITHER EXPRESS OR IMPLIED, AND
 * LICENSOR HEREBY DISCLAIMS ALL SUCH WARRANTIES, INCLUDING WITHOUT
 * LIMITATION, ANY WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
 * PURPOSE, QUIET ENJOYMENT, OR NON-INFRINGEMENT. See the RPL for specific
 * language governing rights and limitations under the RPL.
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

