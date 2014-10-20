#ifndef QUICKRANK_UTILS_RADIX_H_
#define QUICKRANK_UTILS_RADIX_H_

#include <memory>

/*! \file radix.hpp
 * \brief Set of functions implementing descending radix sort for floating point values (ideal for long array)
 */



inline unsigned int flip(unsigned int x) { return x^(-int(x>>31)|0x80000000); } //!<flip a float for sorting: if it's negative, it flips all bits otherwise flips the sign only
inline unsigned int iflip(unsigned int x) { return x^(((x>>31)-1)|0x80000000); } //!<flip a float back (invert flip)

template<typename T>
std::unique_ptr<unsigned int[]> idx_radixsort(T const* fvalues, const unsigned int nvalues) {
  unsigned int *ivalues = new unsigned int[nvalues];
  // TODO: (by cla) The following was not working with mac compiler
  unsigned int* lbucket = new unsigned int[65536] (); // unsigned int lbucket[65536] {0};
  unsigned int* hbucket = new unsigned int[65536] (); // unsigned int hbucket[65536] {0};
  for(unsigned int i=0; i<nvalues; ++i) {
    //feature values are flipped so at to be possible to apply radix sort
    unsigned int flippedvalue = flip(*(unsigned int*)(fvalues+i));
    ivalues[i] = flippedvalue,
    ++lbucket[flippedvalue&0xFFFF],
    ++hbucket[flippedvalue>>16];
  }
  // prefixsum on histograms
  for(unsigned int ltmp, htmp, lsum=-1, hsum=-1, i=0; i<65536; ++i) {
    ltmp = lbucket[i], htmp = hbucket[i];
    lbucket[i] = lsum, hbucket[i] = hsum;
    lsum += ltmp, hsum += htmp;
  }
  // final pass
  struct pair_t { unsigned int value, id; } *aux = new pair_t[nvalues];
  for(unsigned int i=0; i<nvalues; ++i)
    aux[++lbucket[ivalues[i]&0xFFFF]] = { ivalues[i], i };
  for(unsigned int i=0; i<nvalues; ++i)
    ivalues[++hbucket[aux[i].value>>16]] = aux[i].id;
  delete [] aux;
  delete [] lbucket;
  delete [] hbucket;
  return std::unique_ptr<unsigned int[]>(ivalues);
}

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

