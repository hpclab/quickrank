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

#include "optimization/post_learning/cleaver/quality_loss_pruning.h"

#include "utils/strutils.h"

namespace quickrank {
namespace optimization {
namespace post_learning {
namespace pruning {

/// Returns the pruning method of the algorithm.
Cleaver::PruningMethod QualityLossPruning::pruning_method() const {
  return Cleaver::PruningMethod::QUALITY_LOSS;
}

bool QualityLossPruning::line_search_pre_pruning() const {
  return true;
}

void QualityLossPruning::pruning(std::set<unsigned int> &pruned_estimators,
                                 std::shared_ptr<data::Dataset> dataset,
                                 std::shared_ptr<metric::ir::Metric> scorer) {

  size_t num_features = dataset->num_features();
  size_t start_last = num_features - last_estimators_to_optimize_;

  std::vector<MetricScore> metric_scores(last_estimators_to_optimize_);
  std::vector<Score> dataset_score(dataset->num_instances());

  // Score the dataset with initial weights
  score(dataset.get(), &dataset_score[0]);

  Feature *features = dataset->at(0, 0);

  std::vector<Score> new_dataset_score(dataset->num_instances(), 0);

  #pragma omp parallel for firstprivate(new_dataset_score)
  for (size_t f = start_last; f < num_features; ++f) {

    // In place of set the feature weight to 0, score the dataset with the
    // Cleaver score function, and reset back the weight, we optimize the
    // process by computing on the fly the new score based on the original
    // score less the contribute given by the f-th feature...
    for (unsigned int s = 0; s < dataset->num_instances(); ++s) {
      new_dataset_score[s] = dataset_score[s] -
          weights_[f] * features[s * num_features + f];
    }

    metric_scores[f - start_last] =
        scorer->evaluate_dataset(dataset, &new_dataset_score[0]);
  }

  // Find the last metric scores
  std::vector<unsigned int> idx(last_estimators_to_optimize_);
  std::iota(idx.begin(), idx.end(), start_last);

  std::sort(idx.begin(), idx.end(),
            [&metric_scores,&start_last]
                (const unsigned int &a, const unsigned int &b) {
              return metric_scores[a-start_last] > metric_scores[b-start_last];
            });

  for (unsigned int f = 0; f < estimators_to_prune_; ++f) {
    pruned_estimators.insert(idx[f]);
  }
}

}  // namespace cleaver
}  // namespace post_learning
}  // namespace optimization
}  // namespace quickrank
