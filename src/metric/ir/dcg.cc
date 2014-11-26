/*
 * dcg.cpp
 *
 *  Created on: Oct 3, 2014
 *      Author: claudio
 */
#include <cmath>
#include <algorithm>

#include "metric/ir/dcg.h"

#include "utils/qsort.h" // quick sort (for small input)
#include "utils/mergesorter.h" // quick sort (for small input)

namespace quickrank {
namespace metric {
namespace ir {

MetricScore Dcg::compute_dcg(
    const quickrank::data::QueryResults* results) const {
  const unsigned int size = std::min(cutoff(), results->num_results());
  double dcg = 0.0;
//#pragma omp parallel for reduction(+:dcg)
  for (unsigned int i = 0; i < size; ++i)
    dcg += (pow(2.0, results->labels()[i]) - 1.0f) / log2(i + 2.0f);
  return (MetricScore) dcg;
}

MetricScore Dcg::evaluate_result_list(const quickrank::data::QueryResults* rl,
                                      const Score* scores) const {
  if (rl->num_results() == 0)
    return 0.0;
  // sort candidadate labels
  //std::unique_ptr<Label[]> sorted_labels = qsort_ext<Label, Score>(rl->labels(), scores, rl->num_results());
  std::unique_ptr<Label[]> sorted_labels =
      copyextdouble_mergesort<Label, Score>(rl->labels(), scores,
                                            rl->num_results());

  data::QueryResults* sorted_results = new data::QueryResults(
      rl->num_results(), sorted_labels.get(), NULL);
  MetricScore dcg = compute_dcg(sorted_results);
  delete sorted_results;

  return dcg;
}

std::unique_ptr<Jacobian> Dcg::jacobian(
    std::shared_ptr<data::RankedResults> ranked) const {
  const unsigned int size = std::min(cutoff(), ranked->num_results());
  std::unique_ptr<Jacobian> jacobian = std::unique_ptr<Jacobian>(
      new Jacobian(ranked->num_results()));

  for (unsigned int i = 0; i < size; ++i) {
    for (unsigned int j = i + 1; j < ranked->num_results(); ++j) {
      // if the score is the same, non changes occur
      if (ranked->sorted_labels()[ranked->pos_of_rank(i)]
          != ranked->sorted_labels()[ranked->pos_of_rank(j)]) {
        //*p_jacobian =
        jacobian->at(ranked->pos_of_rank(i), ranked->pos_of_rank(j)) =
            (1.0f / log2((double) (i + 2)) - 1.0f / log2((double) (j + 2)))
                * (pow(2.0,
                       (double) ranked->sorted_labels()[ranked->pos_of_rank(i)])
                    - pow(
                        2.0,
                        (double) ranked->sorted_labels()[ranked->pos_of_rank(j)]));
      }
      //p_jacobian++;
    }
  }

  return jacobian;
}

std::ostream& Dcg::put(std::ostream& os) const {
  if (cutoff() != Metric::NO_CUTOFF)
    return os << "DCG@" << cutoff();
  else
    return os << "DCG";
}

}  // namespace ir
}  // namespace metric
}  // namespace quickrank
