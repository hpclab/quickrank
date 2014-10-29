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

double Dcg::compute_dcg(double const* labels, const unsigned int nlabels,
                        const unsigned int k) const {
  unsigned int size = std::min(k, nlabels);
  double dcg = 0.0;
#pragma omp parallel for reduction(+:dcg)
  for (unsigned int i = 0; i < size; ++i)
    dcg += (pow(2.0, labels[i]) - 1.0f) / log2(i + 2.0f);
  return dcg;
}

MetricScore Dcg::evaluate_result_list(const ResultList& ql) const {
  if (ql.size == 0)
    return 0.0;
  const unsigned int size = std::min(cutoff(), ql.size);
  return (MetricScore) Dcg::compute_dcg(ql.labels, ql.size, size);
}

MetricScore Dcg::compute_dcg(Label const* labels, const unsigned int nlabels,
                             const unsigned int k) const {
  unsigned int size = std::min(k, nlabels);
  double dcg = 0.0;
#pragma omp parallel for reduction(+:dcg)
  for (unsigned int i = 0; i < size; ++i)
    dcg += (pow(2.0, labels[i]) - 1.0f) / log2(i + 2.0f);
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

  return compute_dcg(sorted_labels.get(), rl->num_results(), cutoff());
}

std::unique_ptr<Jacobian> Dcg::get_jacobian(std::shared_ptr<data::QueryResults> results) const {
  const unsigned int size = std::min(cutoff(), results->num_results());
  std::unique_ptr<Jacobian> changes = std::unique_ptr<Jacobian>(
      new Jacobian(results->num_results()));

#pragma omp parallel for
  for (unsigned int i = 0; i < size; ++i) {
    //get the pointer to the i-th line of matrix
    double *vchanges = changes->vectat(i, i + 1);
    for (unsigned int j = i + 1; j < results->num_results(); ++j) {
      *vchanges++ = (1.0f / log2((double) (i + 2))
          - 1.0f / log2((double) (j + 2)))
          * (pow(2.0, (double) results->labels()[i]) - pow(2.0, (double) results->labels()[j]));
    }
  }

  return changes;

}

std::unique_ptr<Jacobian> Dcg::get_jacobian(const ResultList &ql) const {
  const unsigned int size = std::min(cutoff(), ql.size);
  std::unique_ptr<Jacobian> changes = std::unique_ptr<Jacobian>(
      new Jacobian(ql.size));
#pragma omp parallel for
  for (unsigned int i = 0; i < size; ++i) {
    //get the pointer to the i-th line of matrix
    double *vchanges = changes->vectat(i, i + 1);
    for (unsigned int j = i + 1; j < ql.size; ++j) {
      *vchanges++ = (1.0f / log2((double) (i + 2))
          - 1.0f / log2((double) (j + 2)))
          * (pow(2.0, (double) ql.labels[i]) - pow(2.0, (double) ql.labels[j]));
    }
  }
  return changes;
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
