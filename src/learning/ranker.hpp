#ifndef __RANKER_HPP__
#define __RANKER_HPP__

#include "learning/dpset.hpp"
#include "utils/qsort.hpp"

//enum ranker_t { MART, RANKBOOST, RANKNET, ADARANK, COOR_ASCENT, LAMBDARANK, LAMBDAMART, LISTNET, RANDOM_FOREST };

class ranker {
	protected:
		metricscorer *scorer = NULL;
		dpset *training_set = NULL;
		dpset *validation_set = NULL;
		float training_score = 0.0;
		float validation_bestscore = 0.0;
	public:
		ranker() {}
		ranker(dpset *training_set) : training_set(training_set) {}
		virtual ~ranker() {
			delete validation_set,
			delete training_set;
		}

		virtual float eval_dp(float *const *const features, unsigned int idx) const = 0;
		virtual const char *whoami() const = 0;
		virtual void init() = 0;
		virtual void learn() = 0;
		virtual void write_outputtofile(const char *filename) = 0;
		float compute_score(dpset *samples, metricscorer *scorer) const {
			//NOTE this replaces a "lot" of methods used in lmart, ranker, evaluator
			const unsigned int nrankedlists = samples->get_nrankedlists();
			unsigned int *const rloffsets = samples->get_rloffsets();
			float *const *const featurematrix = samples->get_fmatrix();
			float score = 0.0f;
			#pragma omp parallel for reduction(+:score)
			for(unsigned int i=0; i<nrankedlists; ++i) {
				rnklst rl = samples->get_ranklist(i);
				float scores[rl.size];
				unsigned int currdp = rloffsets[i];
				for(unsigned int j=0; j<rl.size; ++j, ++currdp)
					scores[j] = eval_dp(featurematrix, currdp);
				float *sortedlabels = copyextfloat_qsort(rl.labels, scores, rl.size);
				score += scorer->compute_score(rnklst(rl.size, sortedlabels, rl.id));
				delete[] sortedlabels;
			}
			return score/nrankedlists;
		}
		void set_scorer(metricscorer *ms) { scorer = ms; }
		void set_trainingset(dpset *dps) { training_set = dps; }
		void set_validationset(dpset *dps) { validation_set = dps; }
};

#endif
