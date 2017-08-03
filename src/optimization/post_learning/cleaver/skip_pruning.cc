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

#include <math.h>

#include "optimization/post_learning/cleaver/skip_pruning.h"

namespace quickrank {
namespace optimization {
namespace post_learning {
namespace pruning {

/// Returns the pruning method of the algorithm.
Cleaver::PruningMethod SkipPruning::pruning_method() const {
  return Cleaver::PruningMethod::SKIP;
}

bool SkipPruning::line_search_pre_pruning() const {
  return false;
}

void SkipPruning::pruning(std::set<unsigned int> &pruned_estimators,
                          std::shared_ptr<data::Dataset> dataset,
                          std::shared_ptr<metric::ir::Metric> scorer) {

  size_t num_features = dataset->num_features();
  size_t start_last = num_features - last_estimators_to_optimize_;
  size_t estimators_to_select =
      last_estimators_to_optimize_ - estimators_to_prune_;
  double step = (double) last_estimators_to_optimize_ / estimators_to_select;

  std::set<size_t> selected_estimators;
  for (size_t i = 0; i < estimators_to_select; ++i) {
    selected_estimators.insert((size_t) ceil(step * i + start_last));
  }

  for (size_t f = start_last; f < num_features; ++f) {
    if (!selected_estimators.count(f))
      pruned_estimators.insert(f);
  }
}

}  // namespace cleaver
}  // namespace post_learning
}  // namespace optimization
}  // namespace quickrank
