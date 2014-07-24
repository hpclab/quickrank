#ifndef __DPSET_HPP__
#define __DPSET_HPP__

#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cstdint>
#include <omp.h>

#ifdef SHOWTIMER
#include <sys/stat.h>
double filesize(const char *filename) {
	struct stat st;
	stat(filename, &st);
	return st.st_size/1048576.0;
}
#endif

#include "utils/transpose.hpp" // traspose matrix
#include "utils/strutils.hpp" // split string
#include "utils/bitarray.hpp" // bit array implementation
#include "utils/radix.hpp" // sorters
#include "utils/listqsort.hpp" // sorter for linked lists

#define SKIP_DPDESCRIPTION //comment to store dp descriptions
#define PRESERVE_DPFILEORDER //uncomment to store datapoints in the same order as in the input file. NOTE dplist.push() is not yet efficient, i.e. O(|dplist|), but dplist are usually short
#define INIT_NOFEATURES 50 //>0

struct qlist {
	qlist(unsigned int size, float *labels, int qid) : size(size), labels(labels), qid(qid) {}
	unsigned int const size;
	float const *labels;
	unsigned int const qid;
};

class dp { //each dp is related to a line read from file
	public:
		dp(const float label, const unsigned int nline, const unsigned int initsize=1) :
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
		void ins_feature(const unsigned int fid, const float fval) {
			if(fid>=maxsize) {
				maxsize = 2*fid+1;
				features = (float*)realloc(features, sizeof(float)*maxsize);
			}
			for(unsigned int i=maxfid+1; i<fid; features[i++]=0.0f);
			maxfid = fid>maxfid ? fid : maxfid,
			features[fid] = fval;
		}
		float *get_resizedfeatures(const unsigned int size) {
			if(size>=maxsize and size!=0)
				features = (float*)realloc(features, sizeof(float)*size);
			for(unsigned int i=maxfid+1; i<size; features[i++]=0.0f);
			return features;
		}
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
		dp *next;
	friend class dplist;
	#ifdef PRESERVE_DPFILEORDER
	friend bool operator> (dp &left, dp &right) { return left.nline>right.nline; };
	friend void listqsort<dp>(dp *&begin, dp *end);
	#endif
};

//dplist collects datapoints having the same qid
class dplist {
	public:
		dplist(const unsigned int qid) : head(NULL), size(0), qid(qid) {}
		void push(dp* x) {
			x->next = head;
			head = x,
			++size;
		}
		void pop() {
			dp* tmp = head;
			head = head->next,
			--size;
			delete tmp;
		}
		dp *front() const {
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
			listqsort<dp>(head);
		}
		#endif
	private:
		dp *head ;
		unsigned int size, qid;
};

class dpcollection {
	public:
		dpcollection() : arr(NULL), arrsize(0), nlists(0) {}
		~dpcollection() {
			for(unsigned int i=0; i<arrsize; ++i)
				delete arr[i];
			free(arr);
		}
		void insert(const unsigned int qid, dp* x) {
			if(qid>=arrsize) {
				unsigned int newsize = 2*qid+1;
				arr = (dplist**)realloc(arr, sizeof(dplist*)*newsize);
				while(arrsize<newsize) arr[arrsize++] = NULL;
			}
			if(arr[qid]==NULL)
				arr[qid] = new dplist(qid),
				++nlists;
			arr[qid]->push(x);
		}
		unsigned int get_nlists() const {
			return nlists;
		}
		dplist **get_lists() {
			if(nlists==0)
				return NULL;
			dplist **ret = new dplist*[nlists];
			for(unsigned int i=0, j=0; i<arrsize; ++i)
				if(arr[i]!=NULL)
					ret[j++] = arr[i];
			return ret;
		}
	private:
		dplist **arr;
		unsigned int arrsize, nlists;
};

class dpset {
	public:
		dpset(const char *filename) {
			FILE *f = fopen(filename, "r");
			if(f) {
				#ifdef SHOWTIMER
				double readingtimer = omp_get_wtime();
				#endif
				const int nth = omp_get_num_procs();
				unsigned int maxfid = INIT_NOFEATURES-1;
				unsigned int linecounter = 0;
				unsigned int th_ndps[nth];
				for(int i=0; i<nth; ++i)
					th_ndps[i] = 0;
				bitarray th_usedfid[nth];
				dpcollection coll;
				#pragma omp parallel num_threads(nth) shared(maxfid, linecounter)
				while(not feof(f)) {
					ssize_t nread;
					size_t linelength = 0;
					char *line = NULL;
					unsigned int nline = 0;
					//lines are read one-at-a-time by threads
					#pragma omp critical
					{ nread = getline(&line, &linelength, f), nline = ++linecounter; }
					//if something is wrong with getline() or line is empty, skip to the next
					if(nread<=0) { free(line); continue; }
					char *token = NULL, *pch = line;
					//skip initial spaces
					while(ISSPC(*pch) && *pch!='\0') ++pch;
					//skip comment line
					if(*pch=='#') { free(line); continue; }
					//each thread get its id to access th_usedfid[], th_ndps[]
					const int ith = omp_get_thread_num();
					//read label (label is a mandatory field)
					if(ISEMPTY(token=read_token(pch))) exit(2);
					//create a new dp for storing the max number of features seen till now
					dp *datapoint = new dp(atof(token), nline, maxfid+1);
					//read qid (qid is a mandatory field)
					unsigned int qid = atou(read_token(pch), "qid:");
					//read a sequence of features, namely (fid,fval) pairs, then the ending description
					while(!ISEMPTY(token=read_token(pch,'#')))
						if(*token=='#') {
							#ifndef SKIP_DPDESCRIPTION
							datapoint->set_description(strdup(++token));
							#endif
							*pch = '\0';
						} else {
							//read a feature (id,val) from token
							unsigned int fid = 0;
							float fval = 0.0f;
							if(sscanf(token, "%u:%f", &fid, &fval)!=2) exit(4);
							//add feature to the current dp
							datapoint->ins_feature(fid, fval),
							//update used featureids
							th_usedfid[ith].set_up(fid),
							//update maxfid (it should be "atomically" managed but its consistency is not a problem)
							maxfid = fid>maxfid ? fid : maxfid;
						}
					//store current sample in trie
					#pragma omp critical
					{ coll.insert(qid, datapoint); }
					//update thread dp counter
					++th_ndps[ith],
					//free mem
					free(line);
				}
				//close input file
				fclose(f);
				#ifdef SHOWTIMER
				readingtimer = omp_get_wtime()-readingtimer;
				double processingtimer = omp_get_wtime();
				#endif
				//merge thread counters and compute the number of features
				for(int i=1; i<nth; ++i)
					th_usedfid[0] |= th_usedfid[i],
					th_ndps[0] += th_ndps[i];
				//make an array of used features ids
				unsigned int nfeatureids = th_usedfid[0].get_upcounter();
				usedfid = th_usedfid[0].get_uparray(nfeatureids);
				//set counters
				ndps = th_ndps[0],
				nrankedlists = coll.get_nlists(),
				nfeatures = usedfid[nfeatureids-1]+1;
				dplist** dplists = coll.get_lists();
				//allocate memory
				#ifndef SKIP_DPDESCRIPTION
				descriptions = (char**)malloc(sizeof(char*)*ndps),
				#endif
				rloffsets = (unsigned int*)malloc(sizeof(unsigned int)*(nrankedlists+1)),
				rlids = (int*)malloc(sizeof(int)*nrankedlists),
				labels = (float*)malloc(sizeof(float)*ndps);
				//compute 'rloffsets' values (i.e. prefixsum dplist sizes) and populate rlids
				for(unsigned int i=0, sum=0, rlsize=0; i<nrankedlists; ++i) {
					rlsize = dplists[i]->get_size(),
					rlids[i] = dplists[i]->get_qid(),
					rloffsets[i] = sum;
					maxrlsize = rlsize>maxrlsize ? rlsize : maxrlsize,
					sum += rlsize;
				}
				rloffsets[nrankedlists] = ndps;
				//populate descriptions (if set), labels, and feature matrix (dp-major order)
				float **tmpfeatures = (float**)malloc(sizeof(float*)*ndps);
				#pragma omp parallel for
				for(unsigned int i=0; i<nrankedlists; ++i) {
					#ifdef PRESERVE_DPFILEORDER
					dplists[i]->sort_bynline();
					#endif
					for(unsigned int j=rloffsets[i]; j<rloffsets[i+1]; ++j) {
						dp *front = dplists[i]->front();
						#ifndef SKIP_DPDESCRIPTION
						descriptions[j] = front->get_description(),
						#endif
						tmpfeatures[j] = front->get_resizedfeatures(nfeatures),
						labels[j] = front->get_label();
						dplists[i]->pop();
					}
				}
				//traspose current feature matrix to get a feature-major order matrix
				features = (float**)malloc(sizeof(float*)*nfeatures);
				for(unsigned int i=0; i<nfeatures; ++i)
					features[i] = (float*)malloc(sizeof(float)*ndps);
				transpose(features, tmpfeatures, ndps, nfeatures);
				for(unsigned int i=0; i<ndps; ++i)
					free(tmpfeatures[i]);
				free(tmpfeatures);
				//delete feature arrays related to skipped featureids and compact the feature matrix
				for(unsigned int i=0, j=0; i<nfeatureids; ++i, ++j) {
					while(j!=usedfid[i])
						free(features[j++]);
					features[i] = features[j];
				}
				nfeatures = nfeatureids,
				features = (float**)realloc(features, sizeof(float*)*nfeatureids);
				//show statistics
				printf("\tfile = '%s'\n\tno. of datapoints = %u\n\tno. of training queries = %u\n\tmax no. of datapoints in a training query = %u\n\tno. of features = %u\n", filename, ndps, nrankedlists, maxrlsize, nfeatures);
				#ifdef SHOWTIMER
				processingtimer = omp_get_wtime()-processingtimer;
				printf("\telapsed time = reading: %.3f seconds (%.2f MB/s, %d threads) + processing: %.3f seconds\n", readingtimer, filesize(filename)/readingtimer, nth, processingtimer);
				#endif
				//free mem from temporary data structures
				delete [] dplists;
			} else exit(5);
		}
		~dpset() {
			if(features) for(unsigned int i=0; i<nfeatures; ++i) free(features[i]);
			#ifndef SKIP_DPDESCRIPTION
			if(descriptions) for(unsigned int i=0; i<ndps; ++i) free(descriptions[i]);
			free(descriptions),
			#endif
			free(rloffsets),
			free(labels),
			free(features),
			free(rlids);
			delete [] usedfid;
		}
		unsigned int get_nfeatures() const {
			return nfeatures;
		}
		unsigned int get_ndatapoints() const {
			return ndps;
		}
		unsigned int get_nrankedlists() const {
			return nrankedlists;
		}
		qlist get_ranklist(unsigned int i) {
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
		float get_label(unsigned int i) const {
			return labels[i];
		}
		unsigned int get_featureid(unsigned int fidx) const {
			return usedfid[fidx];
		}
	private:
		unsigned int nrankedlists = 0, ndps = 0, nfeatures = 0, maxrlsize = 0;
		unsigned int *rloffsets = NULL; //[0..nrankedlists] i-th rankedlist begins at rloffsets[i] and ends at rloffsets[i+1]-1
		float *labels = NULL; //[0..ndps-1]
		float **features = NULL; //[0..maxfid][0..ndps-1]
		int *rlids = NULL; //[0..nrankedlists-1]
		unsigned int *usedfid = NULL; //
		#ifndef SKIP_DPDESCRIPTION
		char **descriptions = NULL; //[0..ndps-1]
		#endif
};

#endif
