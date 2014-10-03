#ifndef QUICKRANK_UTILS_MERGESORTER_H_
#define QUICKRANK_UTILS_MERGESORTER_H_

/*===============================================================================
 * Copyright (c) 2010-2012 University of Massachusetts.  All Rights Reserved.
 *
 * Use of the RankLib package is subject to the terms of the software license set
 * forth in the LICENSE file included with this software, and also available at
 * http://people.cs.umass.edu/~vdang/ranklib_license.html
 *===============================================================================
 */

template <typename T> class List {
	public:
		List(int n) : n(n) { data = new T[n]; }
		~List() { delete [] data; }
		T get(int i) { return data[i]; }
		void set(int i, const T v) { data[i] = v; }
		int size() const { return n; }
	private:
		int n;
		T *data;
};

int *merge(double *list, int *sortedleft, const int leftlength,
           int *sortedright, const int rightlength, const bool asc);

int *_recursivemergesort(double *list, int *idx, const int idxlength,
                         const bool asc);

int *_mergesort(double *list, const int listlength, const bool asc);

void double_mergesort(double *fvalues, const int nvalues, const bool asc=false);

unsigned int *idxdouble_mergesort(double *arr, const int size, const bool asc=false);

double *copyextdouble_mergesort(double const *extarr, double *arr, const int size,
                                const bool asc=false);

#endif
