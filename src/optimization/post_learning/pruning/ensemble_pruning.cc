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
 *  - Salvatore Trani (salvatore.trani@isti.cnr.it)
 */
#include "optimization/post_learning/pruning/ensemble_pruning.h"

#include "data/dataset.h"
#include "metric/ir/metric.h"
#include "learning/ltr_algorithm.h"

#include <fstream>
#include <iomanip>
#include <chrono>
#include <set>
#include <string.h>
#include <math.h>
#include <cassert>
#include <io/svml.h>

namespace quickrank {
namespace optimization {
namespace post_learning {
namespace pruning {

const std::string EnsemblePruning::NAME_ = "EPRUNING";

const std::vector<std::string> EnsemblePruning::pruningMethodNames = {
  "RANDOM", "LOW_WEIGHTS", "SKIP", "LAST", "QUALITY_LOSS", "SCORE_LOSS"
};

EnsemblePruning::EnsemblePruning(double pruning_rate) :
    pruning_rate_(pruning_rate),
    lineSearch_() {
}

EnsemblePruning::EnsemblePruning(double pruning_rate,
                                 std::shared_ptr<learning::linear::LineSearch> lineSearch) :
    pruning_rate_(pruning_rate),
    lineSearch_(lineSearch) {
}

EnsemblePruning::EnsemblePruning(const pugi::xml_document& model) {
  pugi::xml_node model_info = model.child("optimizer").child("info");
  pugi::xml_node model_tree = model.child("optimizer").child("ensemble");

  pruning_rate_ = model_info.child("pruning-rate").text().as_double();

  unsigned int max_feature = 0;
  for (const auto& couple: model_tree.children()) {

    if (strcmp(couple.name(), "tree") == 0) {
      unsigned int feature = couple.child("index").text().as_uint();
      if (feature > max_feature) {
        max_feature = feature;
      }
    }
  }

  estimators_to_prune_ = 0;
  std::vector<double>(max_feature, 0.0).swap(weights_);
  for (const auto& tree: model_tree.children()) {
    if (strcmp(tree.name(), "tree") == 0) {
      unsigned int feature = tree.child("index").text().as_uint();
      double weight = tree.child("weight").text().as_double();
      weights_[feature - 1] = weight;
      if (weight > 0)
        estimators_to_prune_++;
    }
  }
}

std::ostream& EnsemblePruning::put(std::ostream &os) const {
  os << "# Optimizer: " << name() << std::endl
    << "# pruning rate = " << pruning_rate_ << std::endl
    << "# pruning pruning_method = " << EnsemblePruning::getPruningMethod(
      pruning_method())
    << std::endl;
  if (lineSearch_)
    os << "# Line Search Parameters: " << std::endl << *lineSearch_;
  else
    os << "# No Line Search" << std::endl;
  return os << std::endl;
}

void EnsemblePruning::optimize(
    std::shared_ptr<learning::LTR_Algorithm> algo,
    std::shared_ptr<data::Dataset> training_dataset,
    std::shared_ptr<data::Dataset> validation_dataset,
    std::shared_ptr<metric::ir::Metric> metric,
    size_t partial_save,
    const std::string model_filename) {

  auto begin = std::chrono::steady_clock::now();

  if (pruning_rate_ < 1)
    estimators_to_prune_ = (unsigned int) round(
        pruning_rate_ * training_dataset->num_features());
  else {
    estimators_to_prune_ = pruning_rate_;
    if (estimators_to_prune_ >= training_dataset->num_features()) {
      std::cout << "Impossible to prune everything. Quit!" << std::endl;
      return;
    }
  }

  estimators_to_select_ =
      training_dataset->num_features() - estimators_to_prune_;

  // Set all the weights to 1 (and initialize the vector)
  std::vector<double>(training_dataset->num_features(), 1.0).swap(weights_);

  // compute training and validation scores using starting weights
  std::vector<Score> training_score(training_dataset->num_instances());
  score(training_dataset.get(), &training_score[0]);
  auto init_metric_on_training = metric->evaluate_dataset(training_dataset,
                                                     &training_score[0]);

  std::cout << std::endl;
  std::cout << "# Without pruning:" << std::endl;
  std::cout << std::fixed << std::setprecision(4);
  std::cout << "# --------------------------" << std::endl;
  std::cout << "#       training validation" << std::endl;
  std::cout << "# --------------------------" << std::endl;
  std::cout << std::setw(16) << init_metric_on_training;
  if (validation_dataset) {
    std::vector<Score> validation_score(validation_dataset->num_instances());
    score(validation_dataset.get(), &validation_score[0]);
    auto init_metric_on_validation = metric->evaluate_dataset(
        validation_dataset, &validation_score[0]);
    std::cout << std::setw(9) << init_metric_on_validation << std::endl;
  }
  std::cout << std::endl;

  std::set<unsigned int> pruned_estimators;

  // Some pruning methods needs to perform line search before the pruning
  if (line_search_pre_pruning()) {

    if (!lineSearch_) {
      throw std::invalid_argument(std::string(
          "This pruning pruning_method requires line search"));
    }

    if (lineSearch_->get_weigths().empty()) {

      // Need to do the line search pre pruning. The line search model is empty
      std::cout << "# LineSearch pre-pruning:" << std::endl;
      std::cout << "# --------------------------" << std::endl;
      lineSearch_->learn(training_dataset, validation_dataset, metric,
                         partial_save, model_filename);
    } else {
      // The line search pre pruning is already done and the weights are in
      // the model. We just need to load them.
      std::cout << "# LineSearch pre-pruning already done:" << std::endl;
      std::cout << "# --------------------------" << std::endl;
    }

    // Needs to import the line search learned weights into this model
    import_weights_from_line_search(pruned_estimators);
    std::cout << std::endl;
  }

  pruning(pruned_estimators, training_dataset, metric);

  // Set the weights of the pruned features to 0
  for (unsigned int f: pruned_estimators) {
    weights_[f] = 0;
  }

  // Line search post pruning
  if (lineSearch_) {

    // Filter the dataset by deleting the weight-0 features
    std::shared_ptr<data::Dataset> filtered_training_dataset;
    std::shared_ptr<data::Dataset> filtered_validation_dataset;

    filtered_training_dataset = filter_dataset(training_dataset,
                                               pruned_estimators);
    if (validation_dataset)
      filtered_validation_dataset = filter_dataset(validation_dataset,
                                                   pruned_estimators);

    // Run the line search algorithm
    std::cout << "# LineSearch post-pruning:" << std::endl;
    std::cout << "# --------------------------" << std::endl;
    // On each learn call, line search internally resets the weights vector
    lineSearch_->learn(filtered_training_dataset, filtered_validation_dataset,
                       metric, partial_save, model_filename);
    std::cout << std::endl;

    // Needs to import the line search learned weights into this model
    import_weights_from_line_search(pruned_estimators);

  }

  score(training_dataset.get(), &training_score[0]);
  init_metric_on_training = metric->evaluate_dataset(training_dataset,
                                                     &training_score[0]);

  std::cout << "# With pruning:" << std::endl;
  std::cout << std::fixed << std::setprecision(4);
  std::cout << "# --------------------------" << std::endl;
  std::cout << "#       training validation" << std::endl;
  std::cout << "# --------------------------" << std::endl;
  std::cout << std::setw(16) << init_metric_on_training;
  if (validation_dataset) {
    std::vector<Score> validation_score(validation_dataset->num_instances());
    score(validation_dataset.get(), &validation_score[0]);
    auto init_metric_on_validation = metric->evaluate_dataset(
        validation_dataset, &validation_score[0]);
    std::cout << std::setw(9) << init_metric_on_validation << std::endl;
  }

  auto end = std::chrono::steady_clock::now();
  std::chrono::duration<double> elapsed = std::chrono::duration_cast<
      std::chrono::duration<double>>(end - begin);
  std::cout << std::endl;
  std::cout << "# \t Total training time: " << std::setprecision(2) <<
      elapsed.count() << " seconds" << std::endl;
}

std::ofstream& EnsemblePruning::save_model_to_file(std::ofstream &os) const {
  // write optimizer description
  os << "\t<info>" << std::endl;
  os << "\t\t<type>" << name() << "</type>" << std::endl;
  os << "\t\t<pruning-pruning_method>" << getPruningMethod(pruning_method())
     << "</pruning-pruning_method>" << std::endl;
  os << "\t\t<pruning-rate>" << pruning_rate_ << "</pruning-rate>" << std::endl;
  os << "\t</info>" << std::endl;

  os << "\t<ensemble>" << std::endl;
  auto old_precision = os.precision();
  os.setf(std::ios::floatfield, std::ios::fixed);
  for (unsigned int i = 0; i < weights_.size(); i++) {
    os << "\t\t<tree>" << std::endl;
    os << std::setprecision(3);
    os << "\t\t\t<index>" << i + 1 << "</index>" << std::endl;
    os << std::setprecision(std::numeric_limits<Score>::max_digits10);
    os << "\t\t\t<weight>" << weights_[i] << "</weight>" <<
    std::endl;
    os << "\t\t</tree>" << std::endl;
  }
  os << "\t</ensemble>" << std::endl;
  os << std::setprecision(old_precision);
  return os;
}

void EnsemblePruning::score(data::Dataset *dataset, Score *scores) const {

  Feature* features = dataset->at(0,0);
  #pragma omp parallel for
  for (unsigned int s = 0; s < dataset->num_instances(); s++) {
    unsigned int offset_feature = s * dataset->num_features();
    scores[s] = 0;
    // compute partialScore * weight for all the trees
    for (unsigned int f = 0; f < dataset->num_features(); f++) {
      scores[s] += weights_[f] * features[offset_feature + f];
    }
  }
}

void EnsemblePruning::import_weights_from_line_search(
    std::set<unsigned int>& pruned_estimators) {

  std::vector<double> ls_weights = lineSearch_->get_weigths();

  unsigned int ls_f = 0;
  for (unsigned int f = 0; f < weights_.size(); f++) {
    if (!pruned_estimators.count(f)) // skip weights-0 features (pruned by ls)
      weights_[f] = ls_weights[ls_f++];
  }

  assert(ls_f == ls_weights.size());
}

std::shared_ptr<data::Dataset> EnsemblePruning::filter_dataset(
      std::shared_ptr<data::Dataset> dataset,
      std::set<unsigned int>& pruned_estimators) const {

  data::Dataset* filt_dataset = new data::Dataset(dataset->num_instances(),
                                                  estimators_to_select_);

  // allocate feature vector
  std::vector<Feature> featureSelected(estimators_to_select_);
  unsigned int skipped;

  for (unsigned int q = 0; q < dataset->num_queries(); q++) {
    std::shared_ptr<data::QueryResults> results = dataset->getQueryResults(q);
    const Feature* features = results->features();
    const Label* labels = results->labels();

    for (unsigned int r = 0; r < results->num_results(); r++) {
      skipped = 0;
      for (unsigned int f = 0; f < dataset->num_features(); f++) {
        if (pruned_estimators.count(f)) {
          skipped++;
        } else {
          featureSelected[f - skipped] = features[f];
        }
      }
      features += dataset->num_features();
      filt_dataset->addInstance(q, labels[r], featureSelected);
    }
  }

  return std::shared_ptr<data::Dataset>(filt_dataset);
}

}  // namespace pruning
}  // namespace post_learning
}  // namespace optimization
}  // namespace quickrank
