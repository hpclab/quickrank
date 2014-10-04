#ifndef QUICKRANK_METRIC_NDCG_SCORER_H_
#define QUICKRANK_METRIC_NDCG_SCORER_H_

/*! \file ndcgscorer.hpp
 * \brief Normalized Discounted Cumulative Gain (NDCG)
 */

#include <cstdio>
#include <cmath>

#include "metric/metricscorer.h"
#include "utils/strutils.h" // rtnode string
#include "utils/radix.h" // radix sort
#include "utils/qsort.h" // quick sort (for small input)


/*! \class Cache Normalized Discounted Cumulative Gain (nDCG) values.
*/
class ndcgscorer : public DeprecatedMetric {
	public:
		/** Constructor.
		 * @param k maximum number of entities that can be recommended.
		*/
		ndcgscorer(const unsigned int kval) {
			k = kval;
		}
		/** Return a string contatining the name of the metric scorer;
		 */
		const char *whoami() const {
			return "NDCG";
		}
		/* Compute score
		 */
		double compute_score(const qlist &ql);

		/* Compute score
		 */
		fsymmatrix *swap_change(const qlist &ql);
};

#endif
