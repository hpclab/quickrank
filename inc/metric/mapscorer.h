#ifndef QUICKRANK_METRIC_MAP_SCORER_H_
#define QUICKRANK_METRIC_MAP_SCORER_H_

#include <cstdio>

#include "metric/metricscorer.h"
#include "utils/strutils.h" // rtnode string

class MAPScorer : public MetricScorer {
	private:
		unsigned int *relevantdocs, nrelevantdocs;
	public:
		MAPScorer(const unsigned int kval=0) : relevantdocs(NULL), nrelevantdocs(0) {
			printf("\tscorer type = map@%u\n", kval);
			k=kval;
		}
		const char *whoami() const {
			return "MAP";
		}
		double compute_score(const qlist &ql);

		fsymmatrix *swap_change(const qlist &ql);

};

#endif
