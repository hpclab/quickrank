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

#include "metric/ir/dcg.h"

namespace quickrank {
namespace metric {
namespace ir {

const std::string Dcg::NAME_ = "DCG";

MetricScore Dcg::compute_dcg(const Label* labels, unsigned int len) const {
  const unsigned int size = std::min(cutoff(), len);
  double dcg = 0.0;
  for (unsigned int i = 0; i < size; ++i)
    dcg += (pow(2.0, labels[i]) - 1.0f) / log2(i + 2.0f);
  return (MetricScore) dcg;
}

MetricScore Dcg::evaluate_result_list(const quickrank::data::QueryResults* rl,
                                      const Score* scores) const {
  if (rl->num_results() == 0)
    return 0.0;

  // we have at most cutoff to be evaluated
  Label* sorted_l = new Label [cutoff()];
  rl->sorted_labels(scores, sorted_l, cutoff());

  MetricScore dcg = compute_dcg(sorted_l, rl->num_results());

  delete [] sorted_l;

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
    return os << name() << "@" << cutoff();
  else
    return os << name();
}

}  // namespace ir
}  // namespace metric
}  // namespace quickrank
