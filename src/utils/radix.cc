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
#include "utils/radix.h"

static_assert(sizeof(float)==4,"sizeof(float) exception!");
static_assert(sizeof(int)==4,"sizeof(int) exception!");
static_assert(sizeof(size_t)==8,"sizeof(size_t) exception!");

inline size_t flip(size_t x) {
  return x ^ (-int(x >> 31) | 0x80000000);
}  //!<flip a float for sorting: if it's negative, it flips all bits otherwise flips the sign only
inline size_t iflip(size_t x) {
  return x ^ (((x >> 31) - 1) | 0x80000000);
}  //!<flip a float back (invert flip)

std::unique_ptr<size_t[]> idx_radixsort(float const* fvalues,
                                              const size_t nvalues) {
  size_t *ivalues = new size_t[nvalues];
  size_t* lbucket = new size_t[65536]();
  size_t* hbucket = new size_t[65536]();
  for (size_t i = 0; i < nvalues; ++i) {
    //feature values are flipped so at to be possible to apply radix sort
    size_t flippedvalue = flip(*(size_t*) (fvalues + i));
    ivalues[i] = flippedvalue;
    ++lbucket[flippedvalue & 0xFFFF];
    ++hbucket[flippedvalue >> 16];
  }
  // prefixsum on histograms
  for (size_t ltmp, htmp, lsum = -1, hsum = -1, i = 0; i < 65536; ++i) {
    ltmp = lbucket[i];
    htmp = hbucket[i];
    lbucket[i] = lsum;
    hbucket[i] = hsum;
    lsum += ltmp;
    hsum += htmp;
  }
  // final pass
  struct pair_t {
    size_t value, id;
  }*aux = new pair_t[nvalues];
  for (size_t i = 0; i < nvalues; ++i)
    aux[++lbucket[ivalues[i]&0xFFFF]] = {ivalues[i], i};
  for (size_t i = 0; i < nvalues; ++i)
    ivalues[++hbucket[aux[i].value >> 16]] = aux[i].id;
  delete[] aux;
  delete[] lbucket;
  delete[] hbucket;
  return std::unique_ptr<size_t[]>(ivalues);
}

/*! sort an array of float values without modifing the input array and returning permuted indexes of the sorted items
 *  @param fvalues input float array
 *  @param nvalues length of \a fvalues
 *  @return indexes of ascending sorted \a fvalues
 */
size_t *idxfloat_radixsort(float const* fvalues,
                                 const size_t nvalues) {
  size_t *ivalues = new size_t[nvalues];
  // TODO: (by cla) The following was not working with mac compiler
  size_t* lbucket = new size_t[65536]();  // size_t lbucket[65536] {0};
  size_t* hbucket = new size_t[65536]();  // size_t hbucket[65536] {0};
  for (size_t i = 0; i < nvalues; ++i) {
    //feature values are flipped so at to be possible to apply radix sort
    size_t flippedvalue = flip(*(size_t*) (fvalues + i));
    ivalues[i] = flippedvalue, ++lbucket[flippedvalue & 0xFFFF], ++hbucket[flippedvalue
        >> 16];
  }
  // prefixsum on histograms
  for (size_t ltmp, htmp, lsum = -1, hsum = -1, i = 0; i < 65536; ++i) {
    ltmp = lbucket[i], htmp = hbucket[i];
    lbucket[i] = lsum, hbucket[i] = hsum;
    lsum += ltmp, hsum += htmp;
  }
  // final pass
  struct pair_t {
    size_t value, id;
  }*aux = new pair_t[nvalues];
  for (size_t i = 0; i < nvalues; ++i)
    aux[++lbucket[ivalues[i]&0xFFFF]] = {ivalues[i], i};
  for (size_t i = 0; i < nvalues; ++i)
    ivalues[++hbucket[aux[i].value >> 16]] = aux[i].id;
  delete[] aux;
  delete[] lbucket;
  delete[] hbucket;
  return ivalues;
}

/*! sort an array of float values
 *  @param fvalues input float array
 *  @param nvalues length of \a fvalues
 */
template<sortorder const order> void float_radixsort(
    float *fvalues, const size_t nvalues) {
  size_t lbucket[65536] { 0 };
  size_t hbucket[65536] { 0 };
  // histogramming
  for (size_t i = 0; i < nvalues; ++i) {
    size_t x = flip(*(size_t*) (fvalues + i));
    ++lbucket[x & 0xFFFF], ++hbucket[x >> 16];
  }
  // prefixsum on histograms
  if (order == ascending)
    for (size_t ltmp, htmp, lsum = -1, hsum = -1, i = 0; i < 65536; ++i) {
      ltmp = lbucket[i], htmp = hbucket[i];
      lbucket[i] = lsum, hbucket[i] = hsum;
      lsum += ltmp, hsum += htmp;
    }
  else
    for (size_t lsum = nvalues - 1, hsum = nvalues - 1, i = 0; i < 65536;
        ++i) {
      lsum -= lbucket[i], hsum -= hbucket[i];
      lbucket[i] = lsum, hbucket[i] = hsum;
    }
  // final pass
  size_t *aux = new size_t[nvalues];
  for (size_t i = 0; i < nvalues;) {
    size_t x = flip(*(size_t*) (fvalues + i++));
    aux[++lbucket[x & 0xFFFF]] = x;
  }
  for (size_t i = 0; i < nvalues;) {
    size_t x = aux[i++];
    *(size_t*) &fvalues[++hbucket[x >> 16]] = iflip(x);
  }
  delete[] aux;
}

/*! sort an array of float values with respect to another one without modifing the input array and returning permuted indexes of the sorted items
 *  @param extvalues input float array
 *  @param fvalues input float array
 *  @param nvalues length of \a fvalues
 *  @return a sorted copy of \a extvalues wrt \a fvalues
 */
template<sortorder const order> float *copyextfloat_radixsort(
    float const* extvalues, float const* fvalues, const size_t nvalues) {
  size_t lbucket[65536] { 0 };
  size_t hbucket[65536] { 0 };
  for (size_t i = 0; i < nvalues; ++i) {
    //feature values are flipped so at to be possible to apply radix sort
    size_t flippedvalue = flip(*(size_t*) (fvalues + i));
    ++lbucket[flippedvalue & 0xFFFF], ++hbucket[flippedvalue >> 16];
  }
  // prefixsum on histograms
  if (order == ascending)
    for (size_t ltmp, htmp, lsum = -1, hsum = -1, i = 0; i < 65536; ++i) {
      ltmp = lbucket[i], htmp = hbucket[i];
      lbucket[i] = lsum, hbucket[i] = hsum;
      lsum += ltmp, hsum += htmp;
    }
  else
    for (size_t lsum = nvalues - 1, hsum = nvalues - 1, i = 0; i < 65536;
        ++i) {
      lsum -= lbucket[i], hsum -= hbucket[i];
      lbucket[i] = lsum, hbucket[i] = hsum;
    }
  // final pass
  struct pair_t {
    size_t value;
    float extvalue;
  }*aux = new pair_t[nvalues];
  for (size_t i = 0; i < nvalues; ++i) {
    size_t flippedvalue = flip(*(size_t*) (fvalues + i));
    aux[++lbucket[flippedvalue&0xFFFF]] = {flippedvalue, extvalues[i]};
  }
  float *sortedextvalues = new float[nvalues];
  for (size_t i = 0; i < nvalues; ++i)
    sortedextvalues[++hbucket[aux[i].value >> 16]] = aux[i].extvalue;
  delete[] aux;
  return sortedextvalues;
}

