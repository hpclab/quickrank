/*
 * QuickRank - A C++ suite of Learning to Rank algorithms
 * Webpage: http://quickrank.isti.cnr.it/
 * Contact: quickrank@isti.cnr.it
 *
 * Unless explicitly acquired and licensed from Licensor under another
 * license, the contents of this file are subject to the Reciprocal Public
 * License ("RPL") Version 1.5, or subsequent versions as allowed by the RPL,
 * and You may not copy or use this file in either source code or executable
 * form, except in compliance with the terms and conditions of the RPL.
 *
 * All software distributed under the RPL is provided strictly on an "AS
 * IS" basis, WITHOUT WARRANTY OF ANY KIND, EITHER EXPRESS OR IMPLIED, AND
 * LICENSOR HEREBY DISCLAIMS ALL SUCH WARRANTIES, INCLUDING WITHOUT
 * LIMITATION, ANY WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
 * PURPOSE, QUIET ENJOYMENT, OR NON-INFRINGEMENT. See the RPL for specific
 * language governing rights and limitations under the RPL.
 *
 * Contributor:
 *   HPC. Laboratory - ISTI - CNR - http://hpc.isti.cnr.it/
 */
#include <cmath>
#include <algorithm>

#include "metric/ir/tndcg.h"

namespace quickrank {
namespace metric {
namespace ir {

const std::string Tndcg::NAME_ = "TNDCG";

MetricScore Tndcg::compute_tndcg(const quickrank::data::QueryResults* rl,
                                 const Score* scores) const {
  const double idcg = Ndcg::compute_idcg(rl);
  if (idcg <= 0.0)
    return 0;

  unsigned int* idx = new unsigned int [rl->num_results()];
  rl->indexing_of_sorted_labels(scores, idx);

  const unsigned int size = std::min(cutoff(), rl->num_results());
  double tndcg = 0.0;

  for (unsigned int i = 0; i < size;) {
    // find how many with the same score
    // and compute avg score
    double avg_score = pow(2.0, rl->labels()[idx[i]]) - 1.0;
    unsigned int j = i + 1;
    while (j < rl->num_results() && scores[idx[i]] == scores[idx[j]]) {
      avg_score += pow(2.0, rl->labels()[idx[j]]) - 1.0;
      j++;
    }
    avg_score /= (double) (j - i);
    for (unsigned int k = i; k < j; k++)
      tndcg += avg_score / log2(k + 2.0f);

    i = j;
  }

  delete[] idx;
  return (MetricScore) (tndcg / idcg);
}

MetricScore Tndcg::evaluate_result_list(const quickrank::data::QueryResults* rl,
                                        const Score* scores) const {
  if (rl->num_results() == 0)
    return 0.0;

  MetricScore tndcg = compute_tndcg(rl, scores);

  return tndcg;
}

std::unique_ptr<Jacobian> Tndcg::jacobian(
    std::shared_ptr<data::RankedResults> ranked) const {
  std::unique_ptr<Jacobian> jacobian = std::unique_ptr<Jacobian>(
      new Jacobian(ranked->num_results()));

  auto _results = std::shared_ptr<data::QueryResults>(
      new data::QueryResults(ranked->num_results(), ranked->sorted_labels(),
      NULL));
  const double idcg = compute_idcg(_results.get());
  if (idcg <= 0.0)
    return jacobian;

  const unsigned int size = std::min(cutoff(), ranked->num_results());

  double* weights = new double[ranked->num_results()]();  // init with 0s

  /// \todo TODO: it makes sense to pre-compute weights also in ndcg
  for (unsigned int i = 0; i < ranked->num_results();) {
    // find how many with the same score
    // and compute avg score
    unsigned int j = i + 1;
    while (j < ranked->num_results()
        && ranked->sorted_scores()[i] == ranked->sorted_scores()[j])
      j++;

    for (unsigned int k = i; k < j; k++)
      weights[i] += (1.0 / log2(k + 2.0f));
    double tie_size = (double) (j - i);
    weights[i] /= tie_size;     // divide by tie size
    weights[i] /= idcg;         // divide now by idcg to save future operations
    for (unsigned int k = i + 1; k < j; k++)
      weights[k] = weights[i];  // copy for ties
    i = j;
  }

  /// \todo TODO: jacobian->at is expensive, we should do this in the results list order
  /// and not in the re-sorted list
  for (unsigned int i = 0; i < size; ++i) {
    for (unsigned int j = i + 1; j < ranked->num_results(); ++j) {
      // if the score is the same, non changes occur
      if (ranked->sorted_scores()[i] != ranked->sorted_scores()[j]
          && ranked->sorted_labels()[i] != ranked->sorted_labels()[j]) {
        jacobian->at(i, j) = (pow(2.0, ranked->sorted_labels()[i])
            - pow(2.0, ranked->sorted_labels()[j])) * (weights[j] - weights[i]);
      }
    }
  }

  delete[] weights;
  return jacobian;
}

std::ostream& Tndcg::put(std::ostream& os) const {
  if (cutoff() != Metric::NO_CUTOFF)
    return os << name() << "@" << cutoff();
  else
    return os << name();
}

}  // namespace ir
}  // namespace metric
}  // namespace quickrank
