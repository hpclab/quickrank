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

#include "optimization/post_learning/cleaver/last_pruning.h"

namespace quickrank {
namespace optimization {
namespace post_learning {
namespace pruning {

/// Returns the pruning method of the algorithm.
Cleaver::PruningMethod LastPruning::pruning_method() const {
  return Cleaver::PruningMethod::LAST;
}

bool LastPruning::line_search_pre_pruning() const {
  return false;
}

void LastPruning::pruning(std::set<unsigned int> &pruned_estimators,
                          std::shared_ptr<data::Dataset> dataset,
                          std::shared_ptr<metric::ir::Metric> scorer) {

  size_t num_features = (unsigned int) weights_.size();

  for (unsigned int i = 1; i <= estimators_to_prune_; i++) {
    pruned_estimators.insert(num_features - i);
  }
}

}  // namespace cleaver
}  // namespace post_learning
}  // namespace optimization
}  // namespace quickrank
