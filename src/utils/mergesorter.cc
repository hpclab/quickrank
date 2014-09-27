#include "utils/mergesorter.h"

int *merge(double *list, int *sortedleft, const int leftlength, int *sortedright,
           const int rightlength, const bool asc) {
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

int *_recursivemergesort(double *list, int *idx, const int idxlength, const bool asc) {
	if(idxlength == 1) {
		int *dummymerge = new int[1];
		dummymerge[0] = idx[0];
		return dummymerge;
	}

	int mid = idxlength / 2;
	int* left = new int[mid];
	int* right = new int [idxlength-mid];

	for(int i=0; i<mid; ++i)
		left[i] = idx[i];
	for(int i=mid; i<idxlength; ++i)
		right[i-mid] = idx[i];

	int *sortedleft = _recursivemergesort(list, left, mid, asc);
	int *sortedright = _recursivemergesort(list, right, idxlength-mid, asc);

	int* ret = merge(list, sortedleft, mid, sortedright, idxlength-mid, asc);

	delete [] left;
	delete [] right;

	return ret;
}

int *_mergesort(double *list, const int listlength, const bool asc) {
	int* idx = new int [listlength];
	for(int i=0; i<listlength; ++i)
		idx[i] = i;
	int* ret = _recursivemergesort(list, idx, listlength, asc);
	delete [] idx;
	return ret;
}

void double_mergesort(double *fvalues, const int nvalues, const bool asc) {
	int *idx = _mergesort(fvalues, nvalues, asc);
	double* r = new double[nvalues];
	for(int i=0; i<nvalues; ++i)
		r[i] = fvalues[idx[i]];
	for(int i=0; i<nvalues; ++i)
		fvalues[i] = r[i];
	delete [] idx;
	delete [] r;
}

unsigned int *idxdouble_mergesort(double *arr, const int size, const bool asc) {
	int *idx = _mergesort(arr, size, asc);
	unsigned int *uidx = new unsigned int[size];
	for(int i=0; i<size; ++i) uidx[i] = (unsigned int) idx[i];
	delete [] idx;
	return uidx;
}

double *copyextdouble_mergesort(double const *extarr, double *arr, const int size, const bool asc) {
	double *copyof_extarr = new double[size];
	int *idx = _mergesort(arr, size, asc);
	for(int i=0; i<size; ++i)
		copyof_extarr[i] = extarr[idx[i]];
	delete [] idx;
	return copyof_extarr;
}
