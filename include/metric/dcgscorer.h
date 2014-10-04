#ifndef QUICKRANK_METRIC_DCG_SCORER_H_
#define QUICKRANK_METRIC_DCG_SCORER_H_

#include "metric/metricscorer.h"

class DCGScorer : public DeprecatedMetric {
	public:
		DCGScorer(const unsigned int kval) { k = kval; }
		const char *whoami() const { return "DCG"; }
		double compute_score(const qlist &ql);

		fsymmatrix *swap_change(const qlist &ql);

		float get_dcg(unsigned int const* rel, const unsigned int relsize, const unsigned int k);
};

#endif
