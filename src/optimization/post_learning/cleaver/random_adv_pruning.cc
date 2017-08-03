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
#include <limits>

#include "optimization/post_learning/cleaver/random_adv_pruning.h"

namespace quickrank {
namespace optimization {
namespace post_learning {
namespace pruning {

/// Returns the pruning method of the algorithm.
Cleaver::PruningMethod RandomAdvPruning::pruning_method() const {
  return Cleaver::PruningMethod::RANDOM_ADV;
}

bool RandomAdvPruning::line_search_pre_pruning() const {
  return false;
}

void RandomAdvPruning::pruning(std::set<unsigned int> &pruned_estimators,
                               std::shared_ptr<data::Dataset> dataset,
                               std::shared_ptr<metric::ir::Metric> scorer) {

  size_t num_features = weights_.size();
  size_t start_last = num_features - last_estimators_to_optimize_;

  // Score the dataset with the full model
  std::vector<Score> dataset_score(dataset->num_instances());
  score(dataset.get(), &dataset_score[0]);

  Feature *features = dataset->at(0, 0);

  /* initialize random seed: */
  srand(time(NULL));

  MetricScore best_metric_score = std::numeric_limits<double>::lowest();
  auto best_pruned_estimators = std::set<unsigned int>();

  std::vector<Score> new_dataset_score(dataset_score);
  std::set<unsigned int> pruned_estimators_it;

  #pragma omp parallel for firstprivate(new_dataset_score, pruned_estimators_it)
  for (auto i = 0; i < 100; ++i) {

    while (pruned_estimators_it.size() < estimators_to_prune_) {
      size_t index = (rand() % last_estimators_to_optimize_) + start_last;
      if (!pruned_estimators_it.count(index))
        pruned_estimators_it.insert(index);
    }

    for (unsigned int s = 0; s < dataset->num_instances(); ++s) {
      for (auto &f: pruned_estimators_it) {
        new_dataset_score[s] -= weights_[f] * features[s * num_features + f];
      }
    }

    MetricScore metric_scores =
        scorer->evaluate_dataset(dataset, &new_dataset_score[0]);

    #pragma omp critical(metric_update)
    {
      if (metric_scores > best_metric_score) {
        best_metric_score = metric_scores;
        best_pruned_estimators = pruned_estimators_it;
      }
    }
  }

  pruned_estimators = best_pruned_estimators;
}

}  // namespace cleaver
}  // namespace post_learning
}  // namespace optimization
}  // namespace quickrank