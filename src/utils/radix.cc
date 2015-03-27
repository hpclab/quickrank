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
static_assert(sizeof(unsigned int)==4,"sizeof(unsigned int) exception!");


inline unsigned int flip(unsigned int x) {
  return x ^ (-int(x >> 31) | 0x80000000);
}  //!<flip a float for sorting: if it's negative, it flips all bits otherwise flips the sign only
inline unsigned int iflip(unsigned int x) {
  return x ^ (((x >> 31) - 1) | 0x80000000);
}  //!<flip a float back (invert flip)

std::unique_ptr<unsigned int[]> idx_radixsort(float const* fvalues,
                                              const unsigned int nvalues) {
  unsigned int *ivalues = new unsigned int[nvalues];
  unsigned int* lbucket = new unsigned int[65536]();
  unsigned int* hbucket = new unsigned int[65536]();
  for (unsigned int i = 0; i < nvalues; ++i) {
    //feature values are flipped so at to be possible to apply radix sort
    unsigned int flippedvalue = flip(*(unsigned int*) (fvalues + i));
    ivalues[i] = flippedvalue;
    ++lbucket[flippedvalue & 0xFFFF];
    ++hbucket[flippedvalue >> 16];
  }
  // prefixsum on histograms
  for (unsigned int ltmp, htmp, lsum = -1, hsum = -1, i = 0; i < 65536; ++i) {
    ltmp = lbucket[i];
    htmp = hbucket[i];
    lbucket[i] = lsum;
    hbucket[i] = hsum;
    lsum += ltmp;
    hsum += htmp;
  }
  // final pass
  struct pair_t {
    unsigned int value, id;
  }*aux = new pair_t[nvalues];
  for (unsigned int i = 0; i < nvalues; ++i)
    aux[++lbucket[ivalues[i]&0xFFFF]] = {ivalues[i], i};
  for (unsigned int i = 0; i < nvalues; ++i)
    ivalues[++hbucket[aux[i].value >> 16]] = aux[i].id;
  delete[] aux;
  delete[] lbucket;
  delete[] hbucket;
  return std::unique_ptr<unsigned int[]>(ivalues);
}


/*! sort an array of float values without modifing the input array and returning permuted indexes of the sorted items
 *  @param fvalues input float array
 *  @param nvalues length of \a fvalues
 *  @return indexes of ascending sorted \a fvalues
 */
unsigned int *idxfloat_radixsort(float const* fvalues,
                                 const unsigned int nvalues) {
  unsigned int *ivalues = new unsigned int[nvalues];
  // TODO: (by cla) The following was not working with mac compiler
  unsigned int* lbucket = new unsigned int[65536]();  // unsigned int lbucket[65536] {0};
  unsigned int* hbucket = new unsigned int[65536]();  // unsigned int hbucket[65536] {0};
  for (unsigned int i = 0; i < nvalues; ++i) {
    //feature values are flipped so at to be possible to apply radix sort
    unsigned int flippedvalue = flip(*(unsigned int*) (fvalues + i));
    ivalues[i] = flippedvalue, ++lbucket[flippedvalue & 0xFFFF], ++hbucket[flippedvalue
        >> 16];
  }
  // prefixsum on histograms
  for (unsigned int ltmp, htmp, lsum = -1, hsum = -1, i = 0; i < 65536; ++i) {
    ltmp = lbucket[i], htmp = hbucket[i];
    lbucket[i] = lsum, hbucket[i] = hsum;
    lsum += ltmp, hsum += htmp;
  }
  // final pass
  struct pair_t {
    unsigned int value, id;
  }*aux = new pair_t[nvalues];
  for (unsigned int i = 0; i < nvalues; ++i)
    aux[++lbucket[ivalues[i]&0xFFFF]] = {ivalues[i], i};
  for (unsigned int i = 0; i < nvalues; ++i)
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
    float *fvalues, const unsigned int nvalues) {
  unsigned int lbucket[65536] { 0 };
  unsigned int hbucket[65536] { 0 };
  // histogramming
  for (unsigned int i = 0; i < nvalues; ++i) {
    unsigned int x = flip(*(unsigned int*) (fvalues + i));
    ++lbucket[x & 0xFFFF], ++hbucket[x >> 16];
  }
  // prefixsum on histograms
  if (order == ascending)
    for (unsigned int ltmp, htmp, lsum = -1, hsum = -1, i = 0; i < 65536; ++i) {
      ltmp = lbucket[i], htmp = hbucket[i];
      lbucket[i] = lsum, hbucket[i] = hsum;
      lsum += ltmp, hsum += htmp;
    }
  else
    for (unsigned int lsum = nvalues - 1, hsum = nvalues - 1, i = 0; i < 65536;
        ++i) {
      lsum -= lbucket[i], hsum -= hbucket[i];
      lbucket[i] = lsum, hbucket[i] = hsum;
    }
  // final pass
  unsigned int *aux = new unsigned int[nvalues];
  for (unsigned int i = 0; i < nvalues;) {
    unsigned int x = flip(*(unsigned int*) (fvalues + i++));
    aux[++lbucket[x & 0xFFFF]] = x;
  }
  for (unsigned int i = 0; i < nvalues;) {
    unsigned int x = aux[i++];
    *(unsigned int*) &fvalues[++hbucket[x >> 16]] = iflip(x);
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
    float const* extvalues, float const* fvalues, const unsigned int nvalues) {
  unsigned int lbucket[65536] { 0 };
  unsigned int hbucket[65536] { 0 };
  for (unsigned int i = 0; i < nvalues; ++i) {
    //feature values are flipped so at to be possible to apply radix sort
    unsigned int flippedvalue = flip(*(unsigned int*) (fvalues + i));
    ++lbucket[flippedvalue & 0xFFFF], ++hbucket[flippedvalue >> 16];
  }
  // prefixsum on histograms
  if (order == ascending)
    for (unsigned int ltmp, htmp, lsum = -1, hsum = -1, i = 0; i < 65536; ++i) {
      ltmp = lbucket[i], htmp = hbucket[i];
      lbucket[i] = lsum, hbucket[i] = hsum;
      lsum += ltmp, hsum += htmp;
    }
  else
    for (unsigned int lsum = nvalues - 1, hsum = nvalues - 1, i = 0; i < 65536;
        ++i) {
      lsum -= lbucket[i], hsum -= hbucket[i];
      lbucket[i] = lsum, hbucket[i] = hsum;
    }
  // final pass
  struct pair_t {
    unsigned int value;
    float extvalue;
  }*aux = new pair_t[nvalues];
  for (unsigned int i = 0; i < nvalues; ++i) {
    unsigned int flippedvalue = flip(*(unsigned int*) (fvalues + i));
    aux[++lbucket[flippedvalue&0xFFFF]] = {flippedvalue, extvalues[i]};
  }
  float *sortedextvalues = new float[nvalues];
  for (unsigned int i = 0; i < nvalues; ++i)
    sortedextvalues[++hbucket[aux[i].value >> 16]] = aux[i].extvalue;
  delete[] aux;
  return sortedextvalues;
}

