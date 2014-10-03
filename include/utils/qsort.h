#ifndef QUICKRANK_UTILS_QSORT_H_
#define QUICKRANK_UTILS_QSORT_H_

/*! \file qsort.h
 * \brief Set of functions implementing descending quick sort for floating point values (ideal for short array)
 */

/*! sort an array of float values
 *  @param fvalues input float array
 *  @param nvalues length of \a fvalues
 */
void float_qsort(float *arr, const unsigned int size);

/*! sort an array of float values
 *  @param fvalues input float array
 *  @param nvalues length of \a fvalues
 */
void double_qsort(double *arr, const unsigned int size);

/*! sort an array of float values without modifing the input array and returning permuted indexes of the sorted items
 *  @param fvalues input float array
 *  @param nvalues length of \a fvalues
 *  @return indexes of descending sorted \a fvalues
 */
unsigned int *idxfloat_qsort(float const* arr, const unsigned int size);

/*! sort an array of float values without modifing the input array and returning permuted indexes of the sorted items
 *  @param fvalues input float array
 *  @param nvalues length of \a fvalues
 *  @return indexes of descending sorted \a fvalues
 */
unsigned int *idxdouble_qsort(double const* arr, const unsigned int size);

/*! sort an array of float values with respect to another one without modifing the input array and returning permuted indexes of the sorted items
 *  @param extvalues input float array
 *  @param fvalues input float array
 *  @param nvalues length of \a fvalues
 *  @return a sorted copy of \a extvalues wrt \a fvalues
 */
double *copyextdouble_qsort(double const* extarr, double const* arr, const unsigned int size);

/*! sort an array of float values with respect to another one without modifing the input array and returning permuted indexes of the sorted items
 *  @param extvalues input float array
 *  @param fvalues input float array
 *  @param nvalues length of \a fvalues
 *  @return a sorted copy of \a extvalues wrt \a fvalues
 */
float *copyextfloat_qsort(float const* extarr, float const* arr, const unsigned int size);


#endif
