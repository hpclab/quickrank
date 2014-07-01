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
		float *cache = NULL;
		unsigned int cachesize = 0;
		const float undef = -1.0f*FLT_MAX;
	public:
		float operator() (const rnklst &rl, const unsigned int k) {
			if(rl.id>=cachesize || cache[rl.id]==undef) {
				#pragma omp critical
				{
					if(rl.id>=cachesize) {
						unsigned int newcachesize = 2*rl.id+1;
						cache = (float*)realloc(cache,sizeof(float)*newcachesize);
						while(cachesize<newcachesize) cache[cachesize++] = undef;
					}
					cache[rl.id] = compute_idcg(rl.labels, rl.size, k);
				}
			}
			return cache[rl.id];
		}
} idcg_cache;

class ndcgscorer : public metricscorer {
	public:
		ndcgscorer(const unsigned int kval) {
			k = kval;
		}
		const char *whoami() const {
			return "NDCG";
		}
		float compute_score(const rnklst &rl) {
			if(rl.size==0)
				return -1.0f;
			float idcg = idcg_cache(rl, k);
			return idcg>0.0f ? compute_dcg(rl.labels, rl.size, k)/idcg : 0.0f;
		}
		fsymmatrix *swap_change(const rnklst &rl) {
			unsigned int size = k<rl.size ? k : rl.size;
			//compute the ideal ndcg
			float idcg = idcg_cache(rl, k);
			fsymmatrix *changes = new fsymmatrix(rl.size);
			if(idcg>0.0f) {
				#pragma omp parallel for
				for(unsigned int i=0; i<size; ++i) {
					float *vchanges = changes->vectat(i, i+1);
					for(unsigned int j=i+1; j<rl.size; ++j)
						*vchanges++ = (1.0f/log2(i+2.0f)-1.0f/log2(j+2.0f))*(POWEROFTWO(rl.labels[i])-POWEROFTWO(rl.labels[j]))/idcg;
				}
			}
			return changes;
		}
};

#undef POWEROFTWO

#endif
