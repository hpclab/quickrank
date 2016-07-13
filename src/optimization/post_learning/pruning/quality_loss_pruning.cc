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

#include "optimization/post_learning/pruning/quality_loss_pruning.h"

namespace quickrank {
namespace optimization {
namespace post_learning {
namespace pruning {

/// Returns the pruning method of the algorithm.
EnsemblePruning::PruningMethod QualityLossPruning::pruning_method() const {
  return EnsemblePruning::PruningMethod::QUALITY_LOSS;
}

bool QualityLossPruning::line_search_pre_pruning() const {
  return true;
}

void QualityLossPruning::pruning(std::set<unsigned int>& pruned_estimators,
                                    std::shared_ptr<data::Dataset> dataset,
                                    std::shared_ptr<metric::ir::Metric> scorer) {

  unsigned int num_features = dataset->num_features();

  std::vector<MetricScore> metric_scores(num_features);
  std::vector<Score> dataset_score(dataset->num_instances());

  for (unsigned int f = 0; f < num_features; f++) {
    // set the weight of the feature to 0 to simulate its deletion
    double weight_bkp = weights_[f];
    weights_[f] = 0;

    score(dataset.get(), &dataset_score[0]);
    metric_scores[f] = scorer->evaluate_dataset(dataset, &dataset_score[0]);

    // Re set the original weight to the feature
    weights_[f] = weight_bkp;
  }

  // Find the last metric scores
  std::vector<unsigned int> idx (num_features);
  std::iota(idx.begin(), idx.end(), 0);
  std::sort(idx.begin(), idx.end(),
            [&metric_scores] (const unsigned int& a, const unsigned int& b) {
              return metric_scores[a] > metric_scores[b];
            });

  for (unsigned int f = 0; f < estimators_to_prune_; f++) {
    pruned_estimators.insert(idx[f]);
  }
}


}  // namespace pruning
}  // namespace post_learning
}  // namespace optimization
}  // namespace quickrank
