#ifndef QUICKRANK_UTILS_RADIX_H_
#define QUICKRANK_UTILS_RADIX_H_

/*! \file radix.hpp
 * \brief Set of functions implementing descending radix sort for floating point values (ideal for long array)
 */

/*! sort an array of float values without modifing the input array and returning permuted indexes of the sorted items
 *  @param fvalues input float array
 *  @param nvalues length of \a fvalues
 *  @return indexes of ascending sorted \a fvalues
 */
unsigned int *idxfloat_radixsort(float const* fvalues, const unsigned int nvalues);

enum sortorder { ascending, descending };

/*! sort an array of float values
 *  @param fvalues input float array
 *  @param nvalues length of \a fvalues
 */
template <sortorder const order> void float_radixsort(float *fvalues, const unsigned int nvalues);

/*! sort an array of float values with respect to another one without modifing the input array and returning permuted indexes of the sorted items
 *  @param extvalues input float array
 *  @param fvalues input float array
 *  @param nvalues length of \a fvalues
 *  @return a sorted copy of \a extvalues wrt \a fvalues
 */
template <sortorder const order> float *copyextfloat_radixsort(float const* extvalues, float const* fvalues, const unsigned int nvalues);

#endif

