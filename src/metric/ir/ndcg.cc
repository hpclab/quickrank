/*
 * ndcg.cpp
 *
 *  Created on: Oct 3, 2014
 *      Author: claudio
 */
#include <cmath>
#include <algorithm>

#include "metric/ir/ndcg.h"

#include "utils/qsort.h" // quick sort (for small input)
#include "utils/mergesorter.h" // quick sort (for small input)

namespace quickrank {
namespace metric {
namespace ir {

MetricScore Ndcg::compute_idcg(const quickrank::data::QueryResults* rl) const {
  //make a copy of lables
  Label* copyoflabels = new Label[rl->num_results()];
  memcpy(copyoflabels, rl->labels(), sizeof(Label) * rl->num_results());
  //sort the copy
  std::sort(copyoflabels, copyoflabels + rl->num_results(),
            std::greater<int>());
  //compute dcg
  data::QueryResults* sorted_results = new data::QueryResults(rl->num_results(),
                                                              copyoflabels,
                                                              NULL);
  MetricScore dcg = compute_dcg(sorted_results);
  //free mem
  delete sorted_results;
  delete[] copyoflabels;
  //return dcg
  return dcg;
}

MetricScore Ndcg::evaluate_result_list(const quickrank::data::QueryResults* rl,
                                       const Score* scores) const {
  if (rl->num_results() == 0)
    return 0.0;
  const double idcg = Ndcg::compute_idcg(rl);
  if (idcg > 0)
    return Dcg::evaluate_result_list(rl, scores) / idcg;
  else
    return 0;
}

std::unique_ptr<Jacobian> Ndcg::jacobian(
    std::shared_ptr<data::RankedResults> ranked) const {
  std::unique_ptr<Jacobian> jacobian = std::unique_ptr<Jacobian>(
      new Jacobian(ranked->num_results()));

  auto results = std::shared_ptr<data::QueryResults>(
      new data::QueryResults(ranked->num_results(), ranked->sorted_labels(),
      NULL));
  const double idcg = compute_idcg(results.get());
  if (idcg <= 0.0)
    return jacobian;

  const unsigned int size = std::min(cutoff(), ranked->num_results());

  for (unsigned int i = 0; i < size; ++i) {
    for (unsigned int j = i + 1; j < ranked->num_results(); ++j) {
      // if the score is the same, non changes occur
      if (ranked->sorted_labels()[i] != ranked->sorted_labels()[j]) {
        //*p_jacobian =
        jacobian->at(i, j) = (1.0f / log2((double) (j + 2))
            - 1.0f / log2((double) (i + 2)))
            * (pow(2.0, (double) ranked->sorted_labels()[i])
                - pow(2.0, (double) ranked->sorted_labels()[j])) / idcg;
      }
      //p_jacobian++;
    }
  }

  return jacobian;
}

std::ostream& Ndcg::put(std::ostream& os) const {
  if (cutoff() != Metric::NO_CUTOFF)
    return os << "NDCG@" << cutoff();
  else
    return os << "NDCG";
}

}  // namespace ir
}  // namespace metric
}  // namespace quickrank
