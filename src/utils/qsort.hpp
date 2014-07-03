#ifndef __QSORT_HPP__
#define __QSORT_HPP__

/*! \file qsort.hpp
 * \brief Set of functions implementing descending quick sort for floating point values (ideal for short array)
 */

#include <cstring>

template <typename T> void inline swap(T &a, T &b) {
	T tmp = a;
	a = b;
	b = tmp;
}

/*! sort an array of float values
 *  @param fvalues input float array
 *  @param nvalues length of \a fvalues
 */
void float_qsort(float *arr, const unsigned int size) {
	int stack[size];
	int top = 1;
	stack[0] = 0,
	stack[1] = size-1;
	while(top>=0) {
		int h = stack[top--];
		int l = stack[top--];
		float p = arr[h];
		int i = l-1;
		for(int j=l; j<h; ++j)
			if(p<arr[j])
				swap(arr[++i], arr[j]);
		swap(arr[++i], arr[h]);
 		if(i-1>l) {
			stack[++top] = l;
			stack[++top] = i-1;
		}
		if(i+1<h) {
			stack[++top] = i+1;
			stack[++top] = h;
		}
	}
}

/*! sort an array of float values without modifing the input array and returning permuted indexes of the sorted items
 *  @param fvalues input float array
 *  @param nvalues length of \a fvalues
 *  @return indexes of descending sorted \a fvalues
 */
unsigned int *idxfloat_qsort(float const* arr, const unsigned int size) {
	unsigned int *idxarr = new unsigned int[size];
	for(unsigned int i=0; i<size; ++i) idxarr[i] = i;
	float *copyof_arr = new float[size];
	memcpy(copyof_arr, arr, sizeof(float)*size);
	int stack[size];
	int top = 1;
	stack[0] = 0,
	stack[1] = size-1;
	while(top>=0) {
		int h = stack[top--];
		int l = stack[top--];
		float p = copyof_arr[h];
		int i = l-1;
		for(int j=l; j<h; ++j)
			if(p<copyof_arr[j]) {
				swap(copyof_arr[++i], copyof_arr[j]);
				swap(idxarr[i], idxarr[j]);
			}
		swap(copyof_arr[++i], copyof_arr[h]);
		swap(idxarr[i], idxarr[h]);
 		if(i-1>l) {
			stack[++top] = l;
			stack[++top] = i-1;
		}
		if(i+1<h) {
			stack[++top] = i+1;
			stack[++top] = h;
		}
	}
	delete [] copyof_arr;
	return idxarr;
}

/*! sort an array of float values with respect to another one without modifing the input array and returning permuted indexes of the sorted items
 *  @param extvalues input float array
 *  @param fvalues input float array
 *  @param nvalues length of \a fvalues
 *  @return a sorted copy of \a extvalues wrt \a fvalues
 */
float *copyextfloat_qsort(float const* extarr, float const* arr, const unsigned int size) {
	float *copyof_arr = new float[size];
	memcpy(copyof_arr, arr, sizeof(float)*size);
	float *copyof_extarr = new float[size];
	memcpy(copyof_extarr, extarr, sizeof(float)*size);
	int stack[size];
	int top = 1;
	stack[0] = 0,
	stack[1] = size-1;
	while(top>=0) {
		int h = stack[top--];
		int l = stack[top--];
		float p = copyof_arr[h];
		int i = l-1;
		for(int j=l; j<h; ++j)
			if(p<copyof_arr[j]) {
				swap(copyof_arr[++i], copyof_arr[j]);
				swap(copyof_extarr[i], copyof_extarr[j]);
			}
		swap(copyof_arr[++i], copyof_arr[h]);
		swap(copyof_extarr[i], copyof_extarr[h]);
 		if(i-1>l) {
			stack[++top] = l;
			stack[++top] = i-1;
		}
		if(i+1<h) {
			stack[++top] = i+1;
			stack[++top] = h;
		}
	}
	delete [] copyof_arr;
	return copyof_extarr;
}

#endif
