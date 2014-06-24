#ifndef __TRANSPOSE_HPP__
#define __TRANSPOSE_HPP__

#define TRNSP_BLOCKSIZE 64

void transpose(float **output, float **input, const unsigned int nrows, const unsigned int ncols) {
	#pragma omp parallel for
	for(unsigned int rbegin=0; rbegin<nrows; rbegin+=TRNSP_BLOCKSIZE) {
		if(rbegin+TRNSP_BLOCKSIZE<nrows) {
			for(unsigned int cbegin=0; cbegin<ncols; cbegin+=TRNSP_BLOCKSIZE)
				if(cbegin+TRNSP_BLOCKSIZE<ncols) {
					for(unsigned int r=0; r<TRNSP_BLOCKSIZE; ++r)
						for(unsigned int c=0; c<TRNSP_BLOCKSIZE; ++c)
							output[cbegin+c][rbegin+r] = input[rbegin+r][cbegin+c];
				} else {
					for(unsigned int r=0; r<TRNSP_BLOCKSIZE; ++r)
						for(unsigned int c=cbegin; c<ncols; ++c)
							output[c][rbegin+r] = input[rbegin+r][c];
				}
		} else {
			for(unsigned int cbegin=0; cbegin<ncols; cbegin+=TRNSP_BLOCKSIZE) {
				unsigned int cend = cbegin+TRNSP_BLOCKSIZE<ncols ? cbegin+TRNSP_BLOCKSIZE : ncols;
				for(unsigned int r=rbegin; r<nrows; ++r)
					for(unsigned int c=cbegin; c<cend; ++c)
						output[c][r] = input[r][c];
			}
		}
	}
}

#undef TRNSP_BLOCKSIZE

#endif

