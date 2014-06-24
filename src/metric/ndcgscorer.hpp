#ifndef __NDCG_SCORER_HPP__
#define __NDCG_SCORER_HPP__

#include <cstdio>
#include <cmath>

#include "metric/metricscorer.hpp"
#include "utils/trie.hpp" // trie data structure
#include "utils/strutils.hpp" // rtnode string
#include "utils/radix.hpp" // radix sort

#define POWEROFTWO(p) ((float)(1<<((int)(p))))

float compute_dcg(float const* labels, const unsigned int nlabels, const unsigned int k) {
	unsigned int size = (k==0 or k>nlabels) ? nlabels : k;
	float dcg = 0.0f;
	#pragma omp parallel for reduction(+:dcg)
	for(unsigned int i=0; i<size; ++i)
		dcg += (POWEROFTWO(labels[i])-1.0f)/log2(i+2.0f);
	return dcg;
}

float compute_idcg(float const* labels, const unsigned int nlabels, const unsigned int k) {
	//make a copy of lables
	float *copyoflabels = new float[nlabels];
	memcpy(copyoflabels, labels, sizeof(float)*nlabels);
	//sort the copy
	float_radixsort<descending>(copyoflabels, nlabels);
	//compute dcg
	float dcg = compute_dcg(copyoflabels, nlabels, k);
	//free mem
	delete[] copyoflabels;
	//return dcg
	return dcg;
}

//cache idcg value computed for a given rnklst
class {
	private:
		class cachedvalue {
			private:
				float value;
			public:
				cachedvalue(const char *junk) {}
				void set(float v) { value = v; }
				float get() const { return value; }
		};
		trie<cachedvalue> cache;
	public:
		float operator() (const rnklst &rl, const unsigned int k) {
			cachedvalue *tmp = cache.lookup(rl.id);
			if(tmp)
				return tmp->get();
			else {
				float result = compute_idcg(rl.labels, rl.size, k);
				#pragma omp critical
				{ cache.insert(rl.id)->set(result); }
				return result;
			}
		}
} idcg_cache;

class ndcgscorer : public metricscorer {
	private:
		class idealgain {
			public:
				idealgain(const char *junk) : nlabels(0), maxnlabels(0), value(0.0f), labels(NULL) {};
				void push(const float label) {
					if(nlabels==maxnlabels) {
						maxnlabels = 2*maxnlabels+1;
						labels = (float*)malloc(sizeof(float)*maxnlabels);
					}
					labels[nlabels++] = label;
				}
				float get_value() const {
					return value;
				}
				void calculate_value(const unsigned int k) {
					if(nlabels==0) return;
					value = compute_idcg(labels, nlabels, k);
					free(labels),
					nlabels = 0,
					maxnlabels = 0;
				}
			private:
				unsigned int nlabels, maxnlabels;
				float value, *labels;
		};
		trie<idealgain> idcg_extjudg;
	public:
		ndcgscorer(const unsigned int kval) {
			k = kval;
		}
		const char *whoami() const { return "NDCG"; }
		void load_judgments(const char *filename) {
			FILE *f = fopen(filename, "r");
			if(f) {
				#pragma omp parallel
				while(not feof(f)) {
					ssize_t nread;
					size_t linelength = 0;
					char *line = NULL;
					#pragma omp critical
					{ nread = getline(&line, &linelength, f); }
					if(nread<=0) { free(line); continue; }
					char *key = NULL, *token = NULL, *pch = line;
					//skip initial spaces
					while(ISSPC(*pch) && *pch!='\0') ++pch;
					//skip comment line
					if(*pch=='#') { free(line); continue; }
					//read key
					if(ISEMPTY(key=read_token(pch))) exit(2);
					for(int i=0; i!=3; ++i) //skip 2 tokens
						if(ISEMPTY(token=read_token(pch))) exit(2);
					//read label
					float label = atof(token);
					#pragma omp critical
					{ idcg_extjudg.insert(key)->push(label); }
					free(line);
				}
				fclose(f);
				idealgain **arr = idcg_extjudg.get_leaves();
				unsigned int arrsize = idcg_extjudg.get_nleaves();
				#pragma omp parallel
				for(size_t i=0; i<arrsize; ++i)
					arr[i]->calculate_value(k);
				delete [] arr;
				printf(">>LOAD RELEVANCE JUDGMENTS:\n\tfile = '%s'\n\tno. of judgments = %u\n", filename, arrsize);
			}
		}
		float compute_score(const rnklst &rl) {
			if(rl.size==0)
				return -1.0f;
			float d = 0.0f;
			if(not idcg_extjudg.is_empty()) {
				//load external judgment
				idealgain *tmp = idcg_extjudg.lookup(rl.id);
				if(tmp) d = tmp->get_value();
			} else d = idcg_cache(rl, k);
			return d>0.0f ? compute_dcg(rl.labels, rl.size, k)/d : 0.0f;
		}
		fsymmatrix *swap_change(const rnklst &rl) {
			unsigned int size = k<rl.size ? k : rl.size;
			//compute the ideal ndcg
			float d = 0.0f;
			if(not idcg_extjudg.is_empty()) {
				idealgain *tmp = idcg_extjudg.lookup(rl.id);
				if(tmp) d = tmp->get_value();
			} else d = idcg_cache(rl, k);
			fsymmatrix *changes = new fsymmatrix(rl.size);
			if(d>0.0f) {
				#pragma omp parallel for
				for(unsigned int i=0; i<size; ++i) {
					float *vchanges = changes->vectat(i, i+1);
					for(unsigned int j=i+1; j<rl.size; ++j)
						*vchanges++ = (1.0f/log2(i+2.0f)-1.0f/log2(j+2.0f))*(POWEROFTWO(rl.labels[i])-POWEROFTWO(rl.labels[j]))/d;
				}
			}
			return changes;
		}
};

#undef POWEROFTWO

#endif
