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
#ifndef QUICKRANK_UTILS_BITARRAY_H_
#define QUICKRANK_UTILS_BITARRAY_H_

/*! \file bitarray.hpp
 * \brief Implementation of a bit array
 */

#include <cstdlib>
#include <cstdint>

/*! \class bitarray
 *  \brief bit array implementation (1 bit per element)
 */
class BitArray {
 public:
  /** \brief default constructor
   */
  BitArray()
      : data(NULL),
        datasize(0) {
  }
  ~BitArray() {
    free(data);
  }
  /** \brief set the \a i-th bit (the data structure is reallocated to store the \a i-th bit if needed)
   */
  void set_up(const unsigned int i);

  /** \brief return true the \a i-th bit is set (no check is made on the size of the array)
   */
  bool is_up(const unsigned int i) const;
  /** \brief return the number of set bit in the array
   */
  unsigned int get_upcounter();

  /** \brief return an array of integers made up of the set bits positions
   */
  unsigned int *get_uparray(const unsigned int n);

  /** \brief compute bitwse OR of two bit arrays and store the result in the left operand
   */
  BitArray& operator|=(const BitArray& other);

 private:
  int32_t *data;
  unsigned int datasize;
};

#endif
