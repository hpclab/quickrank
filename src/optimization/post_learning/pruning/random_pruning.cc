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

#include "optimization/post_learning/pruning/random_pruning.h"

namespace quickrank {
namespace optimization {
namespace post_learning {
namespace pruning {

/// Returns the pruning method of the algorithm.
EnsemblePruning::PruningMethod RandomPruning::pruning_method() const {
  return EnsemblePruning::PruningMethod::RANDOM;
}

bool RandomPruning::line_search_pre_pruning() const {
  return false;
}

void RandomPruning::pruning(std::set<unsigned int>& pruned_estimators,
                            std::shared_ptr<data::Dataset> dataset,
                            std::shared_ptr<metric::ir::Metric> scorer) {

  unsigned int num_features = (unsigned int) weights_.size();

  /* initialize random seed: */
  srand (time(NULL));

  while (pruned_estimators.size() < estimators_to_prune_) {
    unsigned int index = rand() % num_features;
    if (!pruned_estimators.count(index))
      pruned_estimators.insert(index);
  }
}

}  // namespace pruning
}  // namespace post_learning
}  // namespace optimization
}  // namespace quickrank