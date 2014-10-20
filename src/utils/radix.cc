#include "utils/radix.h"

static_assert(sizeof(float)==4,"sizeof(float) exception!");
static_assert(sizeof(int)==4,"sizeof(int) exception!");
static_assert(sizeof(unsigned int)==4,"sizeof(unsigned int) exception!");

/*! sort an array of float values without modifing the input array and returning permuted indexes of the sorted items
 *  @param fvalues input float array
 *  @param nvalues length of \a fvalues
 *  @return indexes of ascending sorted \a fvalues
 */
unsigned int *idxfloat_radixsort(float const* fvalues, const unsigned int nvalues) {
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
	return ivalues;
}

/*! sort an array of float values
 *  @param fvalues input float array
 *  @param nvalues length of \a fvalues
 */
template <sortorder const order> void float_radixsort(float *fvalues, const unsigned int nvalues) {
	unsigned int lbucket[65536] {0};
	unsigned int hbucket[65536] {0};
	// histogramming
	for(unsigned int i=0; i<nvalues; ++i) {
		unsigned int x=flip(*(unsigned int*)(fvalues+i));
		++lbucket[x&0xFFFF],
		++hbucket[x>>16];
	}
	// prefixsum on histograms
	if(order==ascending)
		for(unsigned int ltmp, htmp, lsum=-1, hsum=-1, i=0; i<65536; ++i) {
			ltmp = lbucket[i], htmp = hbucket[i];
			lbucket[i] = lsum, hbucket[i] = hsum;
			lsum += ltmp, hsum += htmp;
		}
	else
		for(unsigned int lsum=nvalues-1, hsum=nvalues-1, i=0; i<65536; ++i) {
			lsum -= lbucket[i], hsum -= hbucket[i];
			lbucket[i] = lsum, hbucket[i] = hsum;
		}
	// final pass
	unsigned int *aux = new unsigned int[nvalues];
	for(unsigned int i=0; i<nvalues; ) {
		unsigned int x = flip(*(unsigned int*)(fvalues+i++));
		aux[++lbucket[x&0xFFFF]] = x;
	}
	for(unsigned int i=0; i<nvalues; ) {
		unsigned int x = aux[i++];
		*(unsigned int*)&fvalues[++hbucket[x>>16]] = iflip(x);
	}
	delete [] aux;
}

/*! sort an array of float values with respect to another one without modifing the input array and returning permuted indexes of the sorted items
 *  @param extvalues input float array
 *  @param fvalues input float array
 *  @param nvalues length of \a fvalues
 *  @return a sorted copy of \a extvalues wrt \a fvalues
 */
template <sortorder const order> float *copyextfloat_radixsort(float const* extvalues, float const* fvalues, const unsigned int nvalues) {
	unsigned int lbucket[65536] {0};
	unsigned int hbucket[65536] {0};
	for(unsigned int i=0; i<nvalues; ++i) {
		//feature values are flipped so at to be possible to apply radix sort
		unsigned int flippedvalue = flip(*(unsigned int*)(fvalues+i));
		++lbucket[flippedvalue&0xFFFF],
		++hbucket[flippedvalue>>16];
	}
	// prefixsum on histograms
	if(order==ascending)
		for(unsigned int ltmp, htmp, lsum=-1, hsum=-1, i=0; i<65536; ++i) {
			ltmp = lbucket[i], htmp = hbucket[i];
			lbucket[i] = lsum, hbucket[i] = hsum;
			lsum += ltmp, hsum += htmp;
		}
	else
		for(unsigned int lsum=nvalues-1, hsum=nvalues-1, i=0; i<65536; ++i) {
			lsum -= lbucket[i], hsum -= hbucket[i];
			lbucket[i] = lsum, hbucket[i] = hsum;
		}
	// final pass
	struct pair_t { unsigned int value; float extvalue; } *aux = new pair_t[nvalues];
	for(unsigned int i=0; i<nvalues; ++i) {
		unsigned int flippedvalue = flip(*(unsigned int*)(fvalues+i));
		aux[++lbucket[flippedvalue&0xFFFF]] = { flippedvalue, extvalues[i] };
	}
	float *sortedextvalues = new float[nvalues];
	for(unsigned int i=0; i<nvalues; ++i)
		sortedextvalues[++hbucket[aux[i].value>>16]] = aux[i].extvalue;
	delete [] aux;
	return sortedextvalues;
}

