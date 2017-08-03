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

#include "optimization/post_learning/cleaver/score_loss_pruning.h"


namespace quickrank {
namespace optimization {
namespace post_learning {
namespace pruning {

/// Returns the pruning method of the algorithm.
Cleaver::PruningMethod ScoreLossPruning::pruning_method() const {
  return Cleaver::PruningMethod::SCORE_LOSS;
}

bool ScoreLossPruning::line_search_pre_pruning() const {
  return true;
}

void ScoreLossPruning::pruning(std::set<unsigned int> &pruned_estimators,
                               std::shared_ptr<data::Dataset> dataset,
                               std::shared_ptr<metric::ir::Metric> scorer) {

  size_t num_features = dataset->num_features();
  size_t start_last = num_features - last_estimators_to_optimize_;
  size_t num_instances = dataset->num_instances();

  std::vector<Score> feature_scores(last_estimators_to_optimize_, 0);
  std::vector<Score> instance_scores(num_instances, 0);

  // compute the score of each instance
  this->score(dataset.get(), &instance_scores[0]);

  Feature *features = dataset->at(0, 0);

  #pragma omp parallel for
  for (size_t f = start_last; f < num_features; ++f) {
    for (size_t s = 0; s < num_instances; ++s) {
      feature_scores[f - start_last] +=
          weights_[f] * features[s * num_features + f] / instance_scores[s];
    }
  }

  // Find the last feature scores
  std::vector<unsigned int> idx(last_estimators_to_optimize_);
  std::iota(idx.begin(), idx.end(), start_last);
  std::sort(idx.begin(), idx.end(),
            [&feature_scores, &start_last]
                (const unsigned int &a, const unsigned int &b) {
              return feature_scores[a-start_last] < feature_scores[b-start_last];
            });

  for (unsigned int f = 0; f < estimators_to_prune_; f++) {
    pruned_estimators.insert(idx[f]);
  }
}

}  // namespace cleaver
}  // namespace post_learning
}  // namespace optimization
}  // namespace quickrank
