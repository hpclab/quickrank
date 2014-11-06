#include <cmath>
#include <algorithm>

#include "metric/ir/tndcg.h"

#include "utils/qsort.h" // quick sort (for small input)
#include "utils/mergesorter.h" // quick sort (for small input)

namespace quickrank {
namespace metric {
namespace ir {

MetricScore Tndcg::compute_tndcg(const quickrank::data::QueryResults* rl, const Score* scores) const {
  const double idcg = Ndcg::compute_idcg(rl);
  if (idcg <= 0.0)
    return 0;

  unsigned int* idx = idxdouble_mergesort(scores, rl->num_results());

  const unsigned int size = std::min(cutoff(), rl->num_results());
  double tndcg = 0.0;

  for (unsigned int i = 0; i < size;) {
    // find how many with the same score
    // and compute avg score
    double avg_score = pow(2.0,rl->labels()[idx[i]]) -1.0;
    unsigned int j=i+1;
    while (scores[idx[i]]==scores[idx[j]] && j<size) {
      avg_score += pow(2.0,rl->labels()[idx[j]]) -1.0;
      j++;
    }
    avg_score /= (double)(j-i);
    for (unsigned int k=i; k<j; k++)
      tndcg += avg_score / log2(k + 2.0f);

    i=j;
  }

  delete [] idx;
  return (MetricScore) (tndcg/idcg);
}

MetricScore Tndcg::evaluate_result_list(const quickrank::data::QueryResults* rl,
                                       const Score* scores) const {
  if (rl->num_results() == 0)
    return 0.0;

  MetricScore tndcg = compute_tndcg(rl, scores);

  return tndcg;
}

std::unique_ptr<Jacobian> Tndcg::get_jacobian(std::shared_ptr<data::QueryResults> results) const {
  const unsigned int size = std::min(cutoff(), results->num_results());
  const double idcg = compute_idcg(results.get());
  std::unique_ptr<Jacobian> changes = std::unique_ptr<Jacobian>(
      new Jacobian(results->num_results()));

  if (idcg > 0.0) {
#pragma omp parallel for
    for (unsigned int i = 0; i < size; ++i) {
      //get the pointer to the i-th line of matrix
      double *vchanges = changes->vectat(i, i + 1);
      for (unsigned int j = i + 1; j < results->num_results(); ++j) {
        *vchanges++ =
            (1.0f / log2((double) (i + 2)) - 1.0f / log2((double) (j + 2)))
                * (pow(2.0, (double) results->labels()[i])
                    - pow(2.0, (double) results->labels()[j])) / idcg;
      }
    }
  }

  return changes;
}

std::ostream& Tndcg::put(std::ostream& os) const {
  if (cutoff() != Metric::NO_CUTOFF)
    return os << "TNDCG@" << cutoff();
  else
    return os << "TNDCG";
}

}  // namespace ir
}  // namespace metric
}  // namespace quickrank
