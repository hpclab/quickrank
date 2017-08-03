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
#pragma once

#include <memory>
#include <set>

#include "data/dataset.h"
#include "metric/ir/metric.h"
#include "learning/ltr_algorithm.h"
#include "learning/linear/line_search.h"
#include "optimization/optimization.h"
#include "optimization/post_learning/post_learning_opt.h"
#include "pugixml/src/pugixml.hpp"

namespace quickrank {
namespace optimization {
namespace post_learning {
namespace pruning {

/// This implements various strategies for pruning ensembles.
/// This optimization algorithm expect the datasets to be in the partial
/// scores format (i.e., a column for each ensemble, with the partial score
/// returned by that ensamble on each document (row of the original dataset)
class Cleaver: public PostLearningOptimization {

 public:

  enum class PruningMethod {
    RANDOM, RANDOM_ADV, LOW_WEIGHTS, SKIP, LAST,
    QUALITY_LOSS, QUALITY_LOSS_ADV, SCORE_LOSS
  };

  Cleaver(double pruning_rate);

  Cleaver(double pruning_rate,
          std::shared_ptr<learning::linear::LineSearch> lineSearch);

  Cleaver(const pugi::xml_document &model);

  /// Returns the name of the optimizer.
  std::string name() const {
    return NAME_;
  }

  /// Returns the pruning method of the algorithm.
  virtual PruningMethod pruning_method() const = 0;

  virtual bool line_search_pre_pruning() const = 0;

  virtual bool need_partial_score_dataset() const {
    return true;
  };

  virtual void pruning(std::set<unsigned int> &pruned_estimators,
                       std::shared_ptr<data::Dataset> dataset,
                       std::shared_ptr<metric::ir::Metric> scorer) = 0;

  void optimize(std::shared_ptr<quickrank::learning::LTR_Algorithm> algo,
                std::shared_ptr<quickrank::data::Dataset> training_dataset,
                std::shared_ptr<quickrank::data::Dataset> validation_dataset,
                std::shared_ptr<quickrank::metric::ir::Metric> metric,
                size_t partial_save,
                const std::string model_filename);

  /// Process the dataset filtering out features with 0-weight
  virtual std::shared_ptr<data::Dataset> filter_dataset(
      std::shared_ptr<data::Dataset> dataset,
      std::set<unsigned int> &pruned_estimators) const;

  /// Return the xml model representing the current object
  virtual pugi::xml_document *get_xml_model() const;

  // Return the pruining rate of the model
  virtual double get_pruning_rate() {
    return pruning_rate_;
  }

  // Set the pruining rate of the model
  virtual void set_pruning_rate(double pruning_rate) {
    pruning_rate_ = pruning_rate;
  }

  virtual void set_update_model(bool update_model) {
    update_model_ = update_model;
  }

  virtual bool get_update_model() {
    return update_model_;
  }

  static const std::vector<std::string> pruningMethodNames;

  static PruningMethod get_pruning_method(std::string name) {
    auto i_item = std::find(pruningMethodNames.cbegin(),
                            pruningMethodNames.cend(),
                            name);
    if (i_item != pruningMethodNames.cend()) {

      return PruningMethod(std::distance(pruningMethodNames.cbegin(), i_item));
    }

    // TODO: Fix return value...
    throw std::invalid_argument("pruning method " + name + " is not valid");
//    return NULL;
  }

  static std::string get_pruning_method(PruningMethod pruningMethod) {
    return pruningMethodNames[static_cast<int>(pruningMethod)];
  }

  /// Returns the learned weights
  virtual std::vector<double> get_weigths() {
    return std::vector<double>(weights_);
  }

  virtual bool update_weights(std::vector<double>& weights);

  // Return the line search model
  virtual std::shared_ptr<learning::linear::LineSearch> get_line_search() {
    return lineSearch_;
  }

  static const std::string NAME_;

  size_t get_last_estimators_to__work_on() const {
    return last_estimators_to_optimize_;
  }

  void set_last_estimators_to_optimize(size_t last_estimators_to_optimize) {
    last_estimators_to_optimize_ = last_estimators_to_optimize;
  }

  MetricScore get_metric_on_training() {
    return metric_on_training_;
  }

  MetricScore get_metric_on_validation() {
    return metric_on_validation_;
  }

 protected:
  double pruning_rate_;
  unsigned int estimators_to_prune_;
  std::shared_ptr<learning::linear::LineSearch> lineSearch_;
  unsigned int last_estimators_to_optimize_;
  bool update_model_;

  MetricScore metric_on_training_;
  MetricScore metric_on_validation_;

  std::vector<double> weights_;

  /// Prints the description of Algorithm, including its parameters
  std::ostream &put(std::ostream &os) const;

  virtual void score(data::Dataset *dataset, Score *scores) const;

  virtual void import_weights_from_line_search(
      std::set<unsigned int> &pruned_estimators);
};

}  // namespace cleaver
}  // namespace post_learning
}  // namespace optimization
}  // namespace quickrank
