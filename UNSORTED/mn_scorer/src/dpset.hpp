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

#include "utils/strutils.hpp" // split string
#include "utils/bitarray.hpp" // bit array implementation

//#define SKIP_DPDESCRIPTION //comment to store dp descriptions

class DataPoint { //each dp is related to a line read from file
	public:
		dp(const float label, const unsigned int qid, const unsigned int nline, const unsigned int initsize=1) :
			qid(qid),
			nline(nline),
			#ifndef SKIP_DPDESCRIPTION
			description(NULL),
			#endif
			maxsize(initsize>0?initsize:1),
			maxfid(0),
			label(label) {
			features = (float*)malloc(sizeof(float)*maxsize);
			features[0] = 0.0f;
		}
		~DataPoint() {
			#ifndef SKIP_DPDESCRIPTION
			if(description) free(description);
			#endif
			if(features) free(features);
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
			if(size>=maxsize and size>0)
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
	public:
		unsigned int qid, nline;
	private:
		#ifndef SKIP_DPDESCRIPTION
		char *description;
		#endif
		unsigned int maxsize, maxfid;
		float label, *features;
};

void dpswap(DataPoint *&a, DataPoint *&b) {
	DataPoint *tmp = a;
	a = b;
	b = tmp;
}

class dparray {
	public:
		dparray() : arr(NULL), size(0), capacity(0) {}
		~dparray() {
			for(unsigned int i=0; i<size; ++i)
				delete arr[i];
			free(arr);
		}
		void insert(DataPoint* datapoint) {
			if(size==capacity) {
				unsigned int newcapacity = 2*capacity+1;
				arr = (DataPoint**)realloc(arr, sizeof(DataPoint*)*newcapacity);
				while(capacity<newcapacity) arr[capacity++] = NULL;
			}
			arr[size++] = datapoint;
		}
		float *get_resizedfeatures(const unsigned int i, const unsigned int size) const {
			return arr[i]->get_resizedfeatures(size);
		}
		unsigned int get_size() const {
			return size;
		}
		void sort_bynline() {
			int stack[size];
			int top = 1;
			stack[0] = 0,
			stack[1] = size-1;
			while(top>=0) {
				int h = stack[top--];
				int l = stack[top--];
				DataPoint *p = arr[h];
				int i = l-1;
				for(int j=l; j<h; ++j)
					if(p->nline > arr[j]->nline)
						dpswap(arr[++i], arr[j]);
				if(arr[++i]->nline > arr[h]->nline)
					dpswap(arr[i], arr[h]);
				if(i-1>l) {
					stack[++top] = l;
					stack[++top] = i-1;
				}
				if(i+1<h) {
					stack[++top] = i+1;
					stack[++top] = h;
				}
			}
		}
		unsigned int get_nline(unsigned int i) const {
			return arr[i]->nline;
		}
	private:
		DataPoint **arr;
		unsigned int size, capacity;
};

class LTR_VerticalDataset {
	public:
		LTR_VerticalDataset(const char *filename) {
			FILE *f = fopen(filename, "r");
			if(f) {
				const int nth = omp_get_num_procs();
				unsigned int maxfid = 1;
				unsigned int linecounter = 0;
				BitArray th_usedfid[nth];
				#pragma omp parallel num_threads(nth) shared(maxfid,linecounter)
				while(not feof(f)) {
					ssize_t nread;
					size_t linelength = 0;
					char *line = NULL;
					unsigned int nline = (unsigned int) -1;
					//lines are read one-at-a-time by threads
					#pragma omp critical
					{ nread = getline(&line, &linelength, f), nline = linecounter++; }
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
					//read qid (qid is a mandatory field)
					unsigned int qid = atou(read_token(pch), "qid:");
					//create a new dp for storing the max number of features seen till now
					DataPoint *datapoint = new DataPoint(atof(token), qid, nline, maxfid+1);
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
					{ dparr.insert(datapoint); }
					//free mem
					free(line);
				}
				//close input file
				fclose(f);
				//merge features
				for(int i=1; i<nth; ++i)
					th_usedfid[0] |= th_usedfid[i];
				//make an array of used feature ids
				unsigned int nfeatureids = th_usedfid[0].get_upcounter();
				unsigned int *usedfids = th_usedfid[0].get_uparray(nfeatureids);
				//set counters
				ndps = dparr.get_size(),
				nfeatures = usedfids[nfeatureids-1]+1;
				//populate descriptions (if set), labels, and feature matrix (dp-major order)
				features = (float**)malloc(sizeof(float*)*ndps);
				#pragma omp parallel for
				for(unsigned int i=0; i<ndps; ++i)
					features[i] = dparr.get_resizedfeatures(i, nfeatures);
				//show statistics
				printf("%s\t%u\t", filename, ndps);
				//free mem from temporary data structures
				delete[] usedfids;
			} else exit(5);
		}
		~LTR_VerticalDataset() {
			free(features);
		}
		unsigned int get_nfeatures() const {
			return nfeatures;
		}
		unsigned int get_ndatapoints() const {
			return ndps;
		}
		float *get_fvector(unsigned int i) const {
			return features[i];
		}
		unsigned int get_nline(unsigned int i) const {
			return dparr.get_nline(i);
		}
	private:
		unsigned int ndps = 0, nfeatures = 0;
		float **features = NULL;
		dparray dparr;
};

#endif
