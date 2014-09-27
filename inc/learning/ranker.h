#ifndef QUICKRANK_LEARNING_RANKER_H_
#define QUICKRANK_LEARNING_RANKER_H_

#include "learning/dpset.h"
#include "metric/metricscorer.h"
#include "utils/qsort.h"

class Ranker {
	protected:
		MetricScorer *scorer = NULL;
		DataPointDataset *training_set = NULL;
		DataPointDataset *validation_set = NULL;
		float training_score = 0.0f;
		float validation_bestscore = 0.0f;
		unsigned int partialsave_niterations = 0;
		char *output_basename = NULL;
	public:
		Ranker() {}
		virtual ~Ranker() {
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
		void set_scorer(MetricScorer *ms) { scorer = ms; }
		void set_trainingset(DataPointDataset *trainingset) { training_set = trainingset; }
		void set_validationset(DataPointDataset *validationset) { validation_set = validationset; }
		void set_partialsave(unsigned int niterations) { partialsave_niterations = niterations; }
		void set_outputfilename(const char *filename) { output_basename = strdup(filename); }
		float compute_score(DataPointDataset *samples, MetricScorer *scorer);
};

#endif
