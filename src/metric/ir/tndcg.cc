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
    while (j<rl->num_results() && scores[idx[i]]==scores[idx[j]]) {
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

std::unique_ptr<Jacobian> Tndcg::jacobian(std::shared_ptr<data::RankedResults> ranked) const {
  std::unique_ptr<Jacobian> jacobian = std::unique_ptr<Jacobian>(
      new Jacobian(ranked->num_results()));

  auto _results = std::shared_ptr<data::QueryResults> (
      new data::QueryResults (ranked->num_results(), ranked->sorted_labels(), NULL) );
  const double idcg = compute_idcg(_results.get());
  if (idcg <= 0.0)
    return jacobian;

  const unsigned int size = std::min(cutoff(), ranked->num_results());

  double* weights = new double[ranked->num_results()] (); // init with 0s

  /// \todo TODO: it makes sense to pre-compute weights also in ndcg
  for (unsigned int i = 0; i < ranked->num_results();) {
    // find how many with the same score
    // and compute avg score
    unsigned int j=i+1;
    while (j<ranked->num_results() && ranked->sorted_scores()[i]==ranked->sorted_scores()[j])
      j++;

    for (unsigned int k=i; k<j; k++)
      weights[i] += ( 1.0 / log2(k + 2.0f) );
    double tie_size = (double)(j-i);
    weights[i] /= tie_size;     // divide by tie size
    weights[i] /= idcg;         // divide now by idcg to save future operations
    for (unsigned int k=i+1; k<j; k++)
      weights[k] = weights[i];  // copy for ties
    i=j;
  }

  /// \todo TODO: jacobian->at is expensive, we should do this in the results list order
  /// and not in the re-sorted list
  for (unsigned int i = 0; i < size; ++i) {
    for (unsigned int j = i + 1; j < ranked->num_results(); ++j) {
      // if the score is the same, non changes occur
      if (ranked->sorted_scores()[i]!=ranked->sorted_scores()[j] &&
          ranked->sorted_labels()[i]!=ranked->sorted_labels()[j] ) {
        jacobian->at(i,j) =
            ( pow(2.0,ranked->sorted_labels()[i]) - pow(2.0,ranked->sorted_labels()[j]) ) *
            (weights[j] - weights[i]);
      }
    }
  }

  delete [] weights;
  return jacobian;
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
