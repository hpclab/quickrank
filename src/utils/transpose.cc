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
void transpose(float **output, float **input, const unsigned int n, const unsigned int m) {
	#pragma omp parallel for
	for(unsigned int rbegin=0; rbegin<n; rbegin+=TRNSP_BLOCKSIZE) {
		if(rbegin+TRNSP_BLOCKSIZE<n) {
			for(unsigned int cbegin=0; cbegin<m; cbegin+=TRNSP_BLOCKSIZE)
				if(cbegin+TRNSP_BLOCKSIZE<m) {
					for(unsigned int r=0; r<TRNSP_BLOCKSIZE; ++r)
						for(unsigned int c=0; c<TRNSP_BLOCKSIZE; ++c)
							output[cbegin+c][rbegin+r] = input[rbegin+r][cbegin+c];
				} else {
					for(unsigned int r=0; r<TRNSP_BLOCKSIZE; ++r)
						for(unsigned int c=cbegin; c<m; ++c)
							output[c][rbegin+r] = input[rbegin+r][c];
				}
		} else {
			for(unsigned int cbegin=0; cbegin<m; cbegin+=TRNSP_BLOCKSIZE) {
				unsigned int cend = cbegin+TRNSP_BLOCKSIZE<m ? cbegin+TRNSP_BLOCKSIZE : m;
				for(unsigned int r=rbegin; r<n; ++r)
					for(unsigned int c=cbegin; c<cend; ++c)
						output[c][r] = input[r][c];
			}
		}
	}
}

#undef TRNSP_BLOCKSIZE


