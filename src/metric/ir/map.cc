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

#include "metric/ir/map.h"

namespace quickrank {
namespace metric {
namespace ir {

const std::string Map::NAME_ = "MAP";

MetricScore Map::evaluate_result_list(const quickrank::data::QueryResults* rl,
                                      const Score* scores) const {
  unsigned int size = std::min(cutoff(), rl->num_results());
  if (size == 0)
    return 0.0;

  MetricScore ap = 0.0f;
  MetricScore count = 0.0f;
  for (unsigned int i = 0; i < size; ++i)
    if (rl->labels()[i] > 0.0f)
      ap += (++count) / (i + 1.0f);
  return count > 0 ? ap / count : 0.0;
}

std::unique_ptr<Jacobian> Map::jacobian(
    std::shared_ptr<data::RankedResults> ranked) const {
  int* labels = new int[ranked->num_results()];  // int labels[ql.size];
  int* relcount = new int[ranked->num_results()];  // int relcount[ql.size];
  MetricScore count = 0;
  for (unsigned int i = 0; i < ranked->num_results(); ++i) {
    if (ranked->sorted_labels()[i] > 0.0f)  //relevant if true
      labels[i] = 1, ++count;
    else
      labels[i] = 0;
    relcount[i] = count;
  }
  // count = (ql.qid<nrelevantdocs && relevantdocs[ql.qid]>count) ? relevantdocs[ql.qid] : count;
  std::unique_ptr<Jacobian> changes = std::unique_ptr<Jacobian>(
      new Jacobian(ranked->num_results()));
  if (count != 0) {
#pragma omp parallel for
    for (unsigned int i = 0; i < ranked->num_results() - 1; ++i)
      for (unsigned int j = i + 1; j < ranked->num_results(); ++j)
        if (labels[i] != labels[j]) {
          const int diff = labels[j] - labels[i];
          MetricScore change = ((relcount[i] + diff) * labels[j]
              - relcount[i] * labels[i]) / (i + 1.0f);
          for (unsigned int k = i + 1; k < j; ++k)
            if (labels[k] > 0)
              change += (relcount[k] + diff) / (k + 1.0f);
          change += (-relcount[j] * diff) / (j + 1.0f);
          changes->at(i, j) = change / count;
        }
  }
  delete[] labels;
  delete[] relcount;
  return changes;

}

std::ostream& Map::put(std::ostream& os) const {
  if (cutoff() != Metric::NO_CUTOFF)
    return os << name() << "@" << cutoff();
  else
    return os << name();
}

}  // namespace ir
}  // namespace metric
}  // namespace quickrank
