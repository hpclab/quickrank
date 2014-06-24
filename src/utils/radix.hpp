#ifndef __RADIX_HPP__
#define __RADIX_HPP__

static_assert(sizeof(float)==4,"sizeof(float) exception!");
static_assert(sizeof(int)==4,"sizeof(int) exception!");
static_assert(sizeof(unsigned int)==4,"sizeof(unsigned int) exception!");

inline unsigned int flip(unsigned int x) { return x^(-int(x>>31)|0x80000000); } //flip a float for sorting: if it's negative, it flips all bits otherwise flips the sign only
inline unsigned int iflip(unsigned int x) { return x^(((x>>31)-1)|0x80000000); } //flip a float back (invert flip)

//return indexes of fvalues[] sorted in ascending order (nan values are pushed at the end of the sorted array and nvalues is updated)
unsigned int *idxnanfloat_radixsort(float const* fvalues, unsigned int &nvalues) {
	unsigned int *ivalues = new unsigned int[nvalues];
	unsigned int lbucket[65536] {0};
	unsigned int hbucket[65536] {0};
	unsigned int ndefvalues = nvalues;
	for(unsigned int i=0; i<nvalues; ++i)
		if(isundf(fvalues[i])) {
			//undf features are translated into 0xFFFFFFFF so as to be placed at the end of the sorted ivalues
			ivalues[i] = 0xFFFFFFFF,
			++lbucket[0xFFFF],
			++hbucket[0xFFFF],
			//no. of valid features is decremented
			--ndefvalues;
		} else {
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
	nvalues = ndefvalues,
	delete [] aux;
	return ivalues;
}

enum sortorder { ascending, descending };

//sort fvalues[]
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

//return indexes of ascending sorted fvalues array
template <sortorder const order> unsigned int *idxfloat_radixsort(float const* fvalues, const unsigned int nvalues) {
	unsigned int *ivalues = new unsigned int[nvalues];
	unsigned int lbucket[65536] {0};
	unsigned int hbucket[65536] {0};
	for(unsigned int i=0; i<nvalues; ++i) {
		//feature values are flipped so at to be possible to apply radix sort
		unsigned int flippedvalue = flip(*(unsigned int*)(fvalues+i));
		ivalues[i] = flippedvalue,
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
	struct pair_t { unsigned int value, id; } *aux = new pair_t[nvalues];
	for(unsigned int i=0; i<nvalues; ++i)
		aux[++lbucket[ivalues[i]&0xFFFF]] = { ivalues[i], i };
	for(unsigned int i=0; i<nvalues; ++i)
		ivalues[++hbucket[aux[i].value>>16]] = aux[i].id;
	delete [] aux;
	return ivalues;
}

//return a sorted copy of extvalues array. sort is done wrt values of fvalues[]
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

//sort extvalues array based on values of fvalues array
template <sortorder const order> void extfloat_radixsort(float *&extvalues, float const* fvalues, const unsigned int nvalues) {
	unsigned int lbucket[65536] {0};
	unsigned int hbucket[65536] {0};
	unsigned int ndefvalues = nvalues;
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
	for(unsigned int i=0; i<nvalues; ++i)
		extvalues[++hbucket[aux[i].value>>16]] = aux[i].extvalue;
	delete [] aux;
}

#endif

