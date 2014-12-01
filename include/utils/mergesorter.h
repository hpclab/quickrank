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

template<typename T>
int *merge(T const *list, int *sortedleft, const int leftlength,
           int *sortedright, const int rightlength, const bool asc);

template<typename T>
int *_recursivemergesort(T const *list, int *idx, const int idxlength,
                         const bool asc);

template<typename T>
int *_mergesort(T const *list, const int listlength, const bool asc);

template<typename T>
void double_mergesort(T *fvalues, const int nvalues, const bool asc = false);

template<typename T>
unsigned int *idxdouble_mergesort(T *arr, const int size,
                                  const bool asc = false);

template<typename EXT, typename SRC>
std::unique_ptr<EXT[]> copyextdouble_mergesort(EXT const *extarr,
                                               SRC const *arr, const int size,
                                               const bool asc = false);

template<typename T>
unsigned int* positions_mergesort(T *arr, const int size,
                                  const bool asc = false);

template<typename T>
int *merge(T const *list, int *sortedleft, const int leftlength,
           int *sortedright, const int rightlength, const bool asc) {
  int *idx = new int[leftlength + rightlength];
  int i = 0;
  int j = 0;
  int c = 0;
  while (i < leftlength && j < rightlength)
    if (asc) {
      if (list[sortedleft[i]] <= list[sortedright[j]])
        idx[c++] = sortedleft[i++];
      else
        idx[c++] = sortedright[j++];
    } else {
      if (list[sortedleft[i]] >= list[sortedright[j]])
        idx[c++] = sortedleft[i++];
      else
        idx[c++] = sortedright[j++];
    }
  for (; i < leftlength; ++i)
    idx[c++] = sortedleft[i];
  for (; j < rightlength; ++j)
    idx[c++] = sortedright[j];
  delete[] sortedleft, delete[] sortedright;
  return idx;
}

template<typename T>
int *_recursivemergesort(T const *list, int *idx, const int idxlength,
                         const bool asc) {
  if (idxlength == 1) {
    int *dummymerge = new int[1];
    dummymerge[0] = idx[0];
    return dummymerge;
  }

  int mid = idxlength / 2;
  int* left = new int[mid];
  int* right = new int[idxlength - mid];

  for (int i = 0; i < mid; ++i)
    left[i] = idx[i];
  for (int i = mid; i < idxlength; ++i)
    right[i - mid] = idx[i];

  int *sortedleft = _recursivemergesort(list, left, mid, asc);
  int *sortedright = _recursivemergesort(list, right, idxlength - mid, asc);

  int* ret = merge(list, sortedleft, mid, sortedright, idxlength - mid, asc);

  delete[] left;
  delete[] right;

  return ret;
}

template<typename T>
int *_mergesort(T const *list, const int listlength, const bool asc) {
  int* idx = new int[listlength];
  for (int i = 0; i < listlength; ++i)
    idx[i] = i;
  int* ret = _recursivemergesort(list, idx, listlength, asc);
  delete[] idx;
  return ret;
}

template<typename T>
void double_mergesort(T *fvalues, const int nvalues, const bool asc) {
  int *idx = _mergesort(fvalues, nvalues, asc);
  T* r = new T[nvalues];
  for (int i = 0; i < nvalues; ++i)
    r[i] = fvalues[idx[i]];
  for (int i = 0; i < nvalues; ++i)
    fvalues[i] = r[i];
  delete[] idx;
  delete[] r;
}

template<typename T>
unsigned int *idxdouble_mergesort(T *arr, const int size, const bool asc) {
  int *idx = _mergesort(arr, size, asc);
  unsigned int *uidx = new unsigned int[size];
  for (int i = 0; i < size; ++i)
    uidx[i] = (unsigned int) idx[i];
  delete[] idx;
  return uidx;
}

template<typename T>
unsigned int *positions_mergesort(T *arr, const int size, const bool asc) {
  int *idx = _mergesort(arr, size, asc);
  unsigned int *positions = new unsigned int[size];
  for (int i = 0; i < size; ++i)
    positions[idx[i]] = (unsigned int) i;
  delete[] idx;
  return positions;
}

template<typename EXT, typename SRC>
std::unique_ptr<EXT[]> copyextdouble_mergesort(EXT const *extarr,
                                               SRC const *arr, const int size,
                                               const bool asc) {
  EXT *copyof_extarr = new EXT[size];
  int *idx = _mergesort(arr, size, asc);
  for (int i = 0; i < size; ++i)
    copyof_extarr[i] = extarr[idx[i]];
  delete[] idx;
  return std::unique_ptr<EXT[]>(copyof_extarr);
}
#endif
