#ifndef __RANKER_HPP__
#define __RANKER_HPP__

#include "learning/dpset.hpp"
#include "utils/qsort.hpp"

class ranker {
	protected:
		metricscorer *scorer = NULL;
		dpset *training_set = NULL;
		dpset *validation_set = NULL;
		float training_score = 0.0f;
		float validation_bestscore = 0.0f;
		unsigned int partialsave_niterations = 0;
		char *output_basename = NULL;
	public:
		ranker() {}
		virtual ~ranker() {
			delete validation_set,
			delete training_set;
			free(output_basename);
		}
		virtual float eval_dp(float *const *const features, unsigned int idx) const = 0; //prediction value to store in a file
		virtual const char *whoami() const = 0;
		virtual void showme() = 0;
		virtual void init() = 0;
		virtual void learn() = 0;
		virtual void write_outputtofile() = 0;
		void set_scorer(metricscorer *ms) { scorer = ms; }
		void set_trainingset(dpset *trainingset) { training_set = trainingset; }
		void set_validationset(dpset *validationset) { validation_set = validationset; }
		void set_partialsave(unsigned int niterations) { partialsave_niterations = niterations; }
		void set_outputfilename(const char *filename) { output_basename = strdup(filename); }
		float compute_score(dpset *samples, metricscorer *scorer) const {
			//NOTE this replaces a "lot" of methods used in lmart, ranker, evaluator
			const unsigned int nrankedlists = samples->get_nrankedlists();
			unsigned int *const rloffsets = samples->get_rloffsets();
			float *const *const featurematrix = samples->get_fmatrix();
			float score = 0.0f;
			#pragma omp parallel for reduction(+:score)
			for(unsigned int i=0; i<nrankedlists; ++i) {
				qlist ql = samples->get_qlist(i);
				float scores[ql.size];
				for(unsigned int j=0, offset=rloffsets[i]; j<ql.size; )
					scores[j++] = eval_dp(featurematrix, offset++);
				float *sortedlabels = copyextfloat_qsort(ql.labels, scores, ql.size);
				score += scorer->compute_score(qlist(ql.size, sortedlabels, ql.qid));
				delete [] sortedlabels;
			}
			return nrankedlists ? score/nrankedlists : 0.0f;
		}
};

#endif
