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
 * Contributors:
 *  - Salvatore Trani(salvatore.trani@isti.cnr.it)
 */

#include <numeric>

#include "optimization/post_learning/cleaver/low_weights_pruning.h"

namespace quickrank {
namespace optimization {
namespace post_learning {
namespace pruning {

/// Returns the pruning method of the algorithm.
Cleaver::PruningMethod LowWeightsPruning::pruning_method() const {
  return Cleaver::PruningMethod::LOW_WEIGHTS;
}

bool LowWeightsPruning::line_search_pre_pruning() const {
  return true;
}

void LowWeightsPruning::pruning(std::set<unsigned int> &pruned_estimators,
                                std::shared_ptr<data::Dataset> dataset,
                                std::shared_ptr<metric::ir::Metric> scorer) {

  size_t start_last = dataset->num_features() - last_estimators_to_optimize_;

  std::vector<unsigned int> idx(last_estimators_to_optimize_);
  std::iota(idx.begin(), idx.end(), start_last);
  std::sort(idx.begin(), idx.end(),
            [this](const unsigned int &a, const unsigned int &b) {
              return this->weights_[a] < this->weights_[b];
            });

  for (unsigned int f = 0; f < estimators_to_prune_; f++) {
    pruned_estimators.insert(idx[f]);
  }
}

}  // namespace cleaver
}  // namespace post_learning
}  // namespace optimization
}  // namespace quickrank
