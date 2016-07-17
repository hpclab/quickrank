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

#include "optimization/post_learning/pruning/score_loss_pruning.h"


namespace quickrank {
namespace optimization {
namespace post_learning {
namespace pruning {

/// Returns the pruning method of the algorithm.
EnsemblePruning::PruningMethod ScoreLossPruning::pruning_method() const {
  return EnsemblePruning::PruningMethod::SCORE_LOSS;
}

bool ScoreLossPruning::line_search_pre_pruning() const {
  return true;
}

void ScoreLossPruning::pruning(std::set<unsigned int> &pruned_estimators,
                               std::shared_ptr<data::Dataset> dataset,
                               std::shared_ptr<metric::ir::Metric> scorer) {

  unsigned int num_features = dataset->num_features();
  unsigned int num_instances = dataset->num_instances();
  std::vector<Score> feature_scores(num_features, 0);
  std::vector<Score> instance_scores(num_instances, 0);

  // compute the per instance score
  this->score(dataset.get(), &instance_scores[0]);

  Feature *features = dataset->at(0, 0);
#pragma omp parallel for
  for (unsigned int s = 0; s < num_instances; s++) {
    unsigned int offset_feature = s * num_features;
    for (unsigned int f = 0; f < num_features; f++) {
      feature_scores[f] +=
          weights_[f] * features[offset_feature + f] / instance_scores[s];
    }
  }

  // Find the last feature scores
  std::vector<unsigned int> idx(num_features);
  std::iota(idx.begin(), idx.end(), 0);
  std::sort(idx.begin(), idx.end(),
            [&feature_scores](const unsigned int &a, const unsigned int &b) {
              return feature_scores[a] < feature_scores[b];
            });

  for (unsigned int f = 0; f < estimators_to_prune_; f++) {
    pruned_estimators.insert(idx[f]);
  }
}

}  // namespace pruning
}  // namespace post_learning
}  // namespace optimization
}  // namespace quickrank
