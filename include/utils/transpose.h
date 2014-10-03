#ifndef QUICKRANK_UTILS_TRANSPOSE_H_
#define QUICKRANK_UTILS_TRANSPOSE_H_

/*! \file transpose.h
 * \brief transpose matrix
 */

/*! traspose \a input float matrix made up of \a n rows and \a m columns block by block
 *  @param input matrix to transpose
 *  @param output trasposed matrix
 *  @param n number of rows of input matrix
 *  @param m number of columns of input matrix
 */
void transpose(float **output, float **input, const unsigned int n, const unsigned int m);

#endif

