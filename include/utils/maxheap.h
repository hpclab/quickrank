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
#ifndef QUICKRANK_UTILS_MAXHEAP_H_
#define QUICKRANK_UTILS_MAXHEAP_H_

/*! \file maxheap.hpp
 * \brief Implementation of max-heap data structure
 */

#include <cstdlib>
#include <cfloat>

/*! \class mahheap
 *  \brief max-heap implementation with key of type float
 */
template<typename val_t> class MaxHeap {
 public:
  /** \brief default constructor
   * @param initsize set the initial size of the data structure if available
   */
  MaxHeap(unsigned int initsize = 0)
      : maxsize(initsize + 2) {
    arr = (item*) malloc(sizeof(item) * maxsize);
    arrsize = 0, arr[0] = item(DBL_MAX);
  }
  ~MaxHeap() {
    free(arr);
  }
  /** \brief return true if heap is empty
   */
  bool is_notempty() const {
    return arrsize != 0;
  }
  /** \brief return numebr of items stored in the heap
   */
  unsigned int get_size() const {
    return arrsize;
  }
  /** \brief push a new element in the heap and resize the data structure if it is full
   * @param key ordering key of the new element
   * @param val value of the new element
   */
  void push(const double &key, const val_t &val) {
    if (++arrsize == maxsize) {
      maxsize = 2 * maxsize + 1;
      arr = (item*) realloc(arr, sizeof(item) * maxsize);
    }
    unsigned int p = arrsize;
    while (key > arr[p >> 1].key) {
      arr[p] = arr[p >> 1];
      p >>= 1;
    }
    arr[p] = item(key, val);
  }
  /** \brief remove the element on the top of the heap, i.e. the element with max key value
   */
  void pop() {
    const item &last = arr[arrsize--];
    unsigned int child, p = 1;
    while (p << 1 <= arrsize) {
      child = p << 1;
      if (child < arrsize && arr[child + 1].key > arr[child].key)
        ++child;
      if (last.key < arr[child].key)
        arr[p] = arr[child];
      else
        break;
      p = child;
    }
    arr[p] = last;
  }
  /** \brief ref to the top element
   */
  val_t &top() const {
    return arr[1].val;
  }
 protected:
  struct item {
    item(double key)
        : key(key) {
    }
    item(double key, val_t val)
        : key(key),
          val(val) {
    }
    double key;
    val_t val;
  };
  item *arr;
  unsigned int arrsize, maxsize;
};

#endif
