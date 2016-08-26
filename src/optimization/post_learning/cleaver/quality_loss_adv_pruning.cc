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

#include "optimization/post_learning/cleaver/quality_loss_adv_pruning.h"

#include "utils/strutils.h"

namespace quickrank {
namespace optimization {
namespace post_learning {
namespace pruning {

/// Returns the pruning method of the algorithm.
Cleaver::PruningMethod QualityLossAdvPruning::pruning_method() const {
  return Cleaver::PruningMethod::QUALITY_LOSS_ADV;
}

bool QualityLossAdvPruning::line_search_pre_pruning() const {
  return true;
}

void QualityLossAdvPruning::pruning(std::set<unsigned int> &pruned_estimators,
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
  for (unsigned int p = 0; p < estimators_to_prune_; ++p) {

    #pragma omp parallel for firstprivate(new_dataset_score)
    for (size_t f = start_last; f < num_features; ++f) {

      if (pruned_estimators.count(f)) {
        metric_scores[f - start_last] = std::numeric_limits<double>::lowest();
        continue;
      }

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

    auto max = std::max_element(metric_scores.cbegin(), metric_scores.cend());
    auto maxIdx = std::distance(metric_scores.cbegin(), max);
    auto f_prune = maxIdx + start_last;

    pruned_estimators.insert(f_prune);
    // Set the new reference scores after the feature removal
    #pragma omp parallel for
    for (unsigned int s = 0; s < dataset->num_instances(); ++s) {
      dataset_score[s] -=
          weights_[f_prune] * features[s * num_features + f_prune];
    }
  };
}

}  // namespace cleaver
}  // namespace post_learning
}  // namespace optimization
}  // namespace quickrank
