#ifndef QUICKRANK_LEARNING_DPSET_H_
#define QUICKRANK_LEARNING_DPSET_H_

#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cstdint>
#ifdef _OPENMP
#include <omp.h>
#else
#include "utils/omp-stubs.h"
#endif

#ifdef SHOWTIMER
#include <sys/stat.h>
double filesize(const char *filename) {
	struct stat st;
	stat(filename, &st);
	return st.st_size/1048576.0;
}
#endif

#include "utils/transpose.h" // traspose matrix
#include "utils/strutils.h" // split string
#include "utils/bitarray.h" // bit array implementation
#include "utils/radix.h" // sorters

#include "utils/listqsort.h" // sorter for linked lists

#define SKIP_DPDESCRIPTION //comment to store dp descriptions
#define PRESERVE_DPFILEORDER //uncomment to store datapoints in the same order as in the input file. NOTE dplist.push() is not yet efficient, i.e. O(|dplist|), but dplist are usually short
#define INIT_NOFEATURES 50 //>0

struct qlist {
	qlist(unsigned int size, double *labels, int qid) : size(size), labels(labels), qid(qid) {}
	unsigned int const size;
	double const *labels;
	unsigned int const qid;
};

class DataPoint { //each dp is related to a line read from file
	public:
		DataPoint(const float label, const unsigned int nline, const unsigned int initsize=1) :
			maxsize(initsize>0?initsize:1),
			maxfid(0),
			nline(nline),
			label(label),
			features(NULL),
			#ifndef SKIP_DPDESCRIPTION
			description(NULL),
			#endif
			next(NULL) {
			features = (float*)malloc(sizeof(float)*maxsize);
			features[0] = 0.0f;
		}

		void ins_feature(const unsigned int fid, const float fval);

		float *get_resizedfeatures(const unsigned int size);

		float get_label() const {
			return label;
		}
		#ifndef SKIP_DPDESCRIPTION
		void set_description(char *str) {
			description = str;
		}
		char *get_description() const {
			return description;
		}
		#endif
	private:
		unsigned int maxsize, maxfid, nline;
		float label, *features;
		#ifndef SKIP_DPDESCRIPTION
		char *description;
		#endif
		DataPoint *next;
	friend class DataPointList;
	#ifdef PRESERVE_DPFILEORDER
	friend bool operator> (DataPoint &left, DataPoint &right) { return left.nline>right.nline; };
	friend void listqsort<DataPoint>(DataPoint *&begin, DataPoint *end);
	#endif
};

//dplist collects datapoints having the same qid
class DataPointList {
	public:
		DataPointList(const unsigned int qid) : head(NULL), size(0), qid(qid) {}
		void push(DataPoint* x) {
			x->next = head;
			head = x,
			++size;
		}
		void pop() {
			DataPoint* tmp = head;
			head = head->next,
			--size;
			delete tmp;
		}
		DataPoint *front() const {
			return head;
		}
		unsigned int get_size() const {
			return size;
		}
		int get_qid() const {
			return qid;
		}
		#ifdef PRESERVE_DPFILEORDER
		void sort_bynline() {
			listqsort<DataPoint>(head);
		}
		#endif
	private:
		DataPoint *head ;
		unsigned int size, qid;
};

class DataPointCollection {
	public:
		DataPointCollection() : arr(NULL), arrsize(0), nlists(0) {}
		~DataPointCollection() {
			for(unsigned int i=0; i<arrsize; ++i)
				delete arr[i];
			free(arr);
		}

		void insert(const unsigned int qid, DataPoint* x);

		unsigned int get_nlists() const {
			return nlists;
		}

		DataPointList **get_lists();

	private:
		DataPointList **arr;
		unsigned int arrsize, nlists;
};

class DataPointDataset {
	public:
		DataPointDataset(const char *filename);

		~DataPointDataset();

		unsigned int get_nfeatures() const {
			return nfeatures;
		}
		unsigned int get_ndatapoints() const {
			return ndps;
		}
		unsigned int get_nrankedlists() const {
			return nrankedlists;
		}
		qlist get_qlist(unsigned int i) const {
			return qlist(rloffsets[i+1]-rloffsets[i], labels+rloffsets[i], rlids[i]);
		}
		float *get_fvector(unsigned int i) const {
			return features[i];
		}
		float **get_fmatrix() const {
			return features;
		}
		unsigned int *get_rloffsets() const {
			return rloffsets;
		}
		void sort_dpbyfeature(unsigned int i, unsigned int *&sorted, unsigned int &sortedsize) {
			sortedsize = ndps;
			sorted = idxfloat_radixsort(features[i], sortedsize);
		}
		double get_label(unsigned int i) const {
			return labels[i];
		}
		unsigned int get_featureid(unsigned int fidx) const {
			return usedfid[fidx];
		}
		unsigned int get_maxrlsize() const {
			return maxrlsize;
		}
	private:
		unsigned int nrankedlists = 0, ndps = 0, nfeatures = 0, maxrlsize = 0;
		unsigned int *rloffsets = NULL; //[0..nrankedlists] i-th rankedlist begins at rloffsets[i] and ends at rloffsets[i+1]-1
		double *labels = NULL; //[0..ndps-1]
		float **features = NULL; //[0..maxfid][0..ndps-1]
		int *rlids = NULL; //[0..nrankedlists-1]
		unsigned int *usedfid = NULL; //
		#ifndef SKIP_DPDESCRIPTION
		char **descriptions = NULL; //[0..ndps-1]
		#endif
};

#endif
