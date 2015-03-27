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

#include "metric/ir/ndcg.h"


namespace quickrank {
namespace metric {
namespace ir {

const std::string Ndcg::NAME_ = "NDCG";

MetricScore Ndcg::compute_idcg(const quickrank::data::QueryResults* rl) const {
  //make a copy of labels
  Label* copyoflabels = new Label[rl->num_results()];
  memcpy(copyoflabels, rl->labels(), sizeof(Label) * rl->num_results());
  //sort the copy
  std::sort(copyoflabels, copyoflabels + rl->num_results(),
            std::greater<int>());
  //compute dcg
  MetricScore dcg = compute_dcg(copyoflabels,rl->num_results());

  delete[] copyoflabels;
  return dcg;
}

MetricScore Ndcg::evaluate_result_list(const quickrank::data::QueryResults* rl,
                                       const Score* scores) const {
  if (rl->num_results() == 0)
    return 0.0;
  const MetricScore idcg = Ndcg::compute_idcg(rl);
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
    return os << name() << "@" << cutoff();
  else
    return os << name();
}

}  // namespace ir
}  // namespace metric
}  // namespace quickrank
