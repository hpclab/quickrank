#ifndef __DCG_SCORER_HPP__
#define __DCG_SCORER_HPP__

#include <cmath>

#include "metric/metricscorer.hpp"

#define POWEROFTWO(p) ((float)(1<<((unsigned int)(p))))

class dcgscorer : public metricscorer {
	public:
		dcgscorer(const unsigned int kval=10) { k = kval; }
		const char *whoami() const { return "DCG"; }
		float compute_score(const rnklst &rl) {
			if(rl.size<=0)
				return -1.0;
			const unsigned int size = (k>rl.size or k<=0) ? rl.size : k;
			float dcg = 0.0f;
			for(unsigned int i=0; i<size; ++i)
				dcg += (POWEROFTWO(rl.labels[i])-1.0f)/log2(i+2.0f);
			return dcg;
		}
		fsymmatrix *swap_change(const rnklst &rl) {
			fsymmatrix *changes = new fsymmatrix(rl.size);
			unsigned int size = rl.size<k ? rl.size : k;
			#pragma omp parallel for
			for(unsigned int i=0; i<size; ++i) {
				float *vchanges = changes->vectat(i,i+1);
				for(unsigned int j=i+1; j<rl.size; ++j)
					*vchanges++ = (1.0f/log2(i+2.0f)-1.0f/log2(j+2.0f))*(POWEROFTWO(rl.labels[i])-POWEROFTWO(rl.labels[j]));
			}
			return changes;
		}
		float get_dcg(unsigned int const* rel, const unsigned int relsize, const unsigned int k) {
			//NOTE is this method still necessary?
			unsigned int size = (k>relsize or k<=0) ? relsize : k;
			float dcg = 0.0f;
			for(unsigned int i=1; i<=size; ++i)
				dcg += (POWEROFTWO(rel[i-1])-1.0f)/log2(i+1.0f);
			return dcg;
		}
};

#undef POWEROFTWO

#endif
