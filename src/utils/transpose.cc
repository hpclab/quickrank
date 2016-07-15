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

#include "utils/transpose.h"

/*! \def TRNSP_BLOCKSIZE
 *  \brief size of a square block of elements to traspose
 */
#define TRNSP_BLOCKSIZE 64

/*! traspose \a input float matrix made up of \a n rows and \a m columns block by block
 *  @param input matrix to transpose
 *  @param output trasposed matrix
 *  @param n number of rows of input matrix
 *  @param m number of columns of input matrix
 */
void transpose(float **output, float **input, const size_t n,
               const size_t m) {
#pragma omp parallel for
  for (size_t rbegin = 0; rbegin < n; rbegin += TRNSP_BLOCKSIZE) {
    if (rbegin + TRNSP_BLOCKSIZE < n) {
      for (size_t cbegin = 0; cbegin < m; cbegin += TRNSP_BLOCKSIZE)
        if (cbegin + TRNSP_BLOCKSIZE < m) {
          for (size_t r = 0; r < TRNSP_BLOCKSIZE; ++r)
            for (size_t c = 0; c < TRNSP_BLOCKSIZE; ++c)
              output[cbegin + c][rbegin + r] = input[rbegin + r][cbegin + c];
        } else {
          for (size_t r = 0; r < TRNSP_BLOCKSIZE; ++r)
            for (size_t c = cbegin; c < m; ++c)
              output[c][rbegin + r] = input[rbegin + r][c];
        }
    } else {
      for (size_t cbegin = 0; cbegin < m; cbegin += TRNSP_BLOCKSIZE) {
        size_t cend =
            cbegin + TRNSP_BLOCKSIZE < m ? cbegin + TRNSP_BLOCKSIZE : m;
        for (size_t r = rbegin; r < n; ++r)
          for (size_t c = cbegin; c < cend; ++c)
            output[c][r] = input[r][c];
      }
    }
  }
}

#undef TRNSP_BLOCKSIZE

