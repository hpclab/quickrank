#ifndef __MERGESORTER_HPP__
#define __MERGESORTER_HPP__

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

int *merge(float *list, int *sortedleft, const int leftlength, int *sortedright, const int rightlength, const bool asc) {
	int *idx = new int[leftlength+rightlength];
	int i=0;
	int j=0;
	int c=0;
	while(i<leftlength && j<rightlength)
		if(asc) {
			if(list[sortedleft[i]] <= list[sortedright[j]])
				idx[c++] = sortedleft[i++];
			else
				idx[c++] = sortedright[j++];
		} else {
			if(list[sortedleft[i]] >= list[sortedright[j]])
				idx[c++] = sortedleft[i++];
			else
				idx[c++] = sortedright[j++];
		}
	for(;i<leftlength; ++i)
		idx[c++] = sortedleft[i];
	for(;j<rightlength; ++j)
		idx[c++] = sortedright[j];
	delete [] sortedleft,
	delete [] sortedright;
	return idx;
}

int *_recursivemergesort(float *list, int *idx, const int idxlength, const bool asc) {
	if(idxlength == 1) {
		int *dummymerge = new int[1];
		dummymerge[0] = idx[0];
		return dummymerge;
	}

	int mid = idxlength / 2;
	int left[mid];
	int right[idxlength-mid];

	for(int i=0; i<mid; ++i)
		left[i] = idx[i];
	for(int i=mid; i<idxlength; ++i)
		right[i-mid] = idx[i];

	int *sortedleft = _recursivemergesort(list, left, mid, asc);
	int *sortedright = _recursivemergesort(list, right, idxlength-mid, asc);

	return merge(list, sortedleft, mid, sortedright, idxlength-mid, asc);
}

int *_mergesort(float *list, const int listlength, const bool asc) {
	int idx[listlength];
	for(int i=0; i<listlength; ++i)
		idx[i] = i;
	return _recursivemergesort(list, idx, listlength, asc);
}

void float_mergesort(float *fvalues, const int nvalues, const bool asc) {
	int *idx = _mergesort(fvalues, nvalues, asc);
	float r[nvalues];
	for(int i=0; i<nvalues; ++i)
		r[i] = fvalues[idx[i]];
	for(int i=0; i<nvalues; ++i)
		fvalues[i] = r[i];
	delete [] idx;
}

unsigned int *idxfloat_mergesort(float *arr, const int size, const bool asc) {
	int *idx = _mergesort(arr, size, asc);
	unsigned int *uidx = new unsigned int[size];
	for(int i=0; i<size; ++i) uidx[i] = (unsigned int) idx[i];
	delete [] idx;
	return uidx;
}

float *copyextfloat_mergesort(float const *extarr, float *arr, const int size, const bool asc) {
	float *copyof_extarr = new float[size];
	int *idx = _mergesort(arr, size, asc);
	for(int i=0; i<size; ++i)
		copyof_extarr[i] = extarr[idx[i]];
	delete [] idx;
	return copyof_extarr;
}

#endif
