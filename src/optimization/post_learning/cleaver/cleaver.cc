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
#include <fstream>
#include <iomanip>
#include <chrono>
#include <set>
#include <string.h>
#include <math.h>
#include <cassert>
#include <io/svml.h>
#include <sstream>
#include <numeric>

#include "utils/strutils.h"

#include "optimization/post_learning/cleaver/cleaver.h"

namespace quickrank {
namespace optimization {
namespace post_learning {
namespace pruning {

const std::string Cleaver::NAME_ = "CLEAVER";

const std::vector<std::string> Cleaver::pruningMethodNames = {
    "RANDOM", "RANDOM_ADV", "LOW_WEIGHTS", "SKIP", "LAST",
    "QUALITY_LOSS", "QUALITY_LOSS_ADV", "SCORE_LOSS"
};

Cleaver::Cleaver(double pruning_rate) :
    pruning_rate_(pruning_rate),
    lineSearch_(),
    last_estimators_to_optimize_(0),
    update_model_(true) {
}

Cleaver::Cleaver(double pruning_rate,
                 std::shared_ptr<learning::linear::LineSearch> lineSearch)
    :
    pruning_rate_(pruning_rate),
    lineSearch_(lineSearch),
    last_estimators_to_optimize_(0),
    update_model_(true) {
}

Cleaver::Cleaver(const pugi::xml_document &model) {
  pugi::xml_node model_info = model.child("optimizer").child("info");
  pugi::xml_node model_tree = model.child("optimizer").child("ensemble");

  pruning_rate_ = model_info.child("pruning-rate").text().as_double();

  // TODO: read the line search parameters if available in the xml
  model.child("optimizer").child("info").set_name("old-info");

  // modify the model in order to create line search model from xml
  model.child("optimizer").child("line-search").set_name("info");
  model.child("optimizer").set_name("ranker");

  lineSearch_ = std::shared_ptr<learning::linear::LineSearch>(
      new learning::linear::LineSearch(model));

  // revert modification
  model.child("ranker").remove_child("optimizer");
  model.child("optimizer").child("info").set_name("line-search");
  model.child("optimizer").child("old-info").set_name("info");

  // Check if this is a full cleaver model or if it contains only the
  // preamble (for models which uses cleaver inside...)
  if (!model_tree.child("tree").child("index").empty()) {

    unsigned int max_feature = 0;
    for (const auto &tree: model_tree.children("tree")) {

      unsigned int feature = tree.child("index").text().as_uint();
      if (feature > max_feature) {
        max_feature = feature;
      }
    }

    estimators_to_prune_ = 0;
    std::vector<double>(max_feature, 0.0).swap(weights_);
    for (const auto &tree: model_tree.children("tree")) {
      unsigned int feature = tree.child("index").text().as_uint();
      float weight = tree.child("weight").text().as_float();
      weights_[feature - 1] = weight;
      if (weight == 0)
        estimators_to_prune_++;
    }
  }
}

pugi::xml_document *Cleaver::get_xml_model() const {

  pugi::xml_document *doc = new pugi::xml_document();
  pugi::xml_node root = doc->append_child("optimizer");

  pugi::xml_node info = root.append_child("info");

  info.append_child("opt-algo").text() = name().c_str();
  info.append_child("opt-method").text() =
      get_pruning_method(pruning_method()).c_str();
  info.append_child("pruning-rate").text() = pruning_rate_;

  if (lineSearch_) {
    pugi::xml_document &ls_model = *lineSearch_->get_xml_model();
    pugi::xml_node ls_info = ls_model.child("ranker").child("info");

    // use the info section of the line search model to add a new node into
    // the xml of the cleaver model
    ls_info.set_name("line-search");
    root.append_copy(ls_info);
  }

  std::stringstream ss;
  ss << std::setprecision(std::numeric_limits<float>::max_digits10);

  pugi::xml_node ensemble = root.append_child("ensemble");
  for (unsigned int i = 0; i < weights_.size(); i++) {
    pugi::xml_node tree = ensemble.append_child("tree");

    ss << weights_[i];

    tree.append_child("index").text() = i + 1;
    tree.append_child("weight").text() = ss.str().c_str();

    // reset ss
    ss.str(std::string());
  }

  return doc;
}


std::ostream &Cleaver::put(std::ostream &os) const {
  os << "# Optimizer: " << name() << std::endl
     << "# pruning rate = " << pruning_rate_ << std::endl
     << "# pruning pruning_method = " << Cleaver::get_pruning_method(
      pruning_method())
     << std::endl;
  if (lineSearch_)
    os << "# Line Search Parameters: " << std::endl << *lineSearch_;
  else
    os << "# No Line Search";
  return os << std::endl;
}

void Cleaver::optimize(
    std::shared_ptr<learning::LTR_Algorithm> algo,
    std::shared_ptr<data::Dataset> training_dataset,
    std::shared_ptr<data::Dataset> validation_dataset,
    std::shared_ptr<metric::ir::Metric> metric,
    size_t partial_save,
    const std::string model_filename) {

  auto begin = std::chrono::steady_clock::now();

  unsigned int num_features = training_dataset->num_features();

  // If var is not set, optimize all the features and not only the last ones.
  bool opt_last_only = true;
  if (last_estimators_to_optimize_ == 0) {
    last_estimators_to_optimize_ = num_features;
    opt_last_only = false;
  }

  if (pruning_rate_ < 1)
    estimators_to_prune_ = round(pruning_rate_ * last_estimators_to_optimize_);
  else {
    estimators_to_prune_ = pruning_rate_;
    if (estimators_to_prune_ >= last_estimators_to_optimize_) {
      std::cerr << "Incorrect pruning rate value (too high). Quit!"
                << std::endl;
      exit(EXIT_FAILURE);
    }
  }

  // If weights were not set before by calling the update_weights method,
  // set the starting weights to the algo weights
  if (weights_.empty()) {
      weights_ = std::vector<double>(algo->get_weights());
  } else if (weights_.size() != num_features) {
    // The check on the number of features is needed because the line search model
    // could be reused on a different datasets (different size) w/o reset weights
    std::cerr << "Initial Cleaver Weights does not correspond to datasets "
        "size" << std::endl;
    exit(EXIT_FAILURE);
  }

  // compute training and validation scores using starting weights
  std::vector<Score> training_score(training_dataset->num_instances());
  score(training_dataset.get(), &training_score[0]);
  metric_on_training_ = metric->evaluate_dataset(training_dataset,
                                                 &training_score[0]);

  std::cout << std::endl;
  std::cout << "# Model before optimization:" << std::endl;
  std::cout << std::fixed << std::setprecision(4);
  std::cout << "# --------------------------" << std::endl;
  std::cout << "#  size training validation" << std::endl;
  std::cout << "# --------------------------" << std::endl;
  std::cout << std::setw(7) << num_features;
  std::cout << std::setw(8) << metric_on_training_;
  if (validation_dataset) {
    std::vector<Score> validation_score(validation_dataset->num_instances());
    score(validation_dataset.get(), &validation_score[0]);
    metric_on_validation_ = metric->evaluate_dataset(
        validation_dataset, &validation_score[0]);
    std::cout << std::setw(10) << metric_on_validation_ << std::endl;
  }
  std::cout << std::endl;

  // Backup the starting weights in order to re-set them to their starting
  // value after the pre-pruning line search (which modifies the weights)
  std::vector<double> starting_weights(weights_);

  // Some pruning methods needs to perform line search before the pruning
  if (line_search_pre_pruning() && estimators_to_prune_ > 0 && lineSearch_) {

    print_weights(weights_, "Cleaver Weights ANTE LS pre-pruning");

    // Set to optimize only last estimators
    if (opt_last_only)
      lineSearch_->set_last_only(last_estimators_to_optimize_);

    if (lineSearch_->get_weights().empty()) {

      // Need to do the line search pre-pruning.
      // The line search weights inside the model are not set

      // Set the starting weights in the line search model...
      bool res = lineSearch_->update_weights(weights_);
      if (!res)
        std::exit(EXIT_FAILURE);

      std::cout << "# LineSearch pre-pruning:" << std::endl;
      std::cout << "# --------------------------" << std::endl;
      lineSearch_->learn(training_dataset, validation_dataset, metric,
                         0, std::string());

      // Needs to import the line search learned weights into this model
      auto ls_weights = lineSearch_->get_weights();
      weights_ = std::vector<double>(ls_weights);

    } else {
      // The line search pre pruning is already done and the weights are in
      // the model. We just need to load them.
      std::cout << "# LineSearch pre-pruning already done:" << std::endl;
      std::cout << "# --------------------------" << std::endl;

      // import weights from line search and scale accordingly
      // to the average weight of the LtR algo (LS done separately has
      // weights around 1.0f value).
      auto ls_weights = lineSearch_->get_weights();
      auto algo_weights = algo->get_weights();

      assert(ls_weights.size() == algo_weights.size());

      // window_size is the mean weight times the window_size_ factor
      double mean_ls_weight = std::accumulate(ls_weights.cbegin(),
                                              ls_weights.cend(),
                                              0.0) / ls_weights.size();

      double mean_algo_weight = std::accumulate(algo_weights.cbegin(),
                                                algo_weights.cend(),
                                                0.0) / algo_weights.size();

      double scaling_factor = mean_ls_weight / mean_algo_weight;
      weights_ = std::vector<double>(ls_weights);
      std::transform(weights_.begin(), weights_.end(), weights_.begin(),
                     std::bind1st(std::multiplies<double>(),
                                  1.0 / scaling_factor) );
    }

    std::cout << std::endl;
    print_weights(weights_, "Cleaver Weights POST LS pre-pruning");
  }

  auto begin_pruning = std::chrono::steady_clock::now();

  std::set<unsigned int> pruned_estimators;
  pruning(pruned_estimators, training_dataset, metric);

  auto end_pruning = std::chrono::steady_clock::now();
  std::chrono::duration<double> pTime = std::chrono::duration_cast<
      std::chrono::duration<double>>(end_pruning - begin_pruning);

  std::cout << "# Ensemble Pruning:" << std::endl;
  std::cout << "# --------------------------" << std::endl;
  std::cout << "# Removed " << pruned_estimators.size() << " out of "
      << num_features << " trees (" << std::setprecision(2) << pTime.count()
      << " s.)" << std::endl << std::endl;

  print_weights(std::vector<double>(
      pruned_estimators.cbegin(),
      pruned_estimators.cend()), "Pruned trees");

  // Reset the weights to their starting value
  weights_ = starting_weights;

  // Set the weights of the pruned features to 0
  for (unsigned int f: pruned_estimators) {
    weights_[f] = 0;
  }

  // Line search post pruning
  if (lineSearch_) {

    print_weights(weights_, "Cleaver Weights PRE LS post-pruning");

    // Re-Set the weights of the LS model for the post-pruning phase
    // Need it because pre-pruning could have modified the weights and because
    // now the dataset is smaller in size (pruned features...)
    std::vector<double> new_ls_weights;
    new_ls_weights.reserve(num_features - estimators_to_prune_);

    for (size_t f=0; f<num_features; ++f) {
      if (!pruned_estimators.count(f)) // skip pruned estimators
        new_ls_weights.push_back(weights_[f]);
    }

    bool res = lineSearch_->update_weights(new_ls_weights);
    if (!res)
      std::exit(EXIT_FAILURE);

    // Filter the dataset by deleting features with 0 weight
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

    // Set to optimize only last estimators (excluding the pruned ones)
    if (opt_last_only) {
      lineSearch_->set_last_only(
          last_estimators_to_optimize_ - estimators_to_prune_);
    }

    // On each call to learn, line search internally resets the weights vector
    lineSearch_->learn(filtered_training_dataset, filtered_validation_dataset,
                       metric, 0, std::string());
    std::cout << std::endl;

    // Needs to import the line search learned weights into this model
    import_weights_from_line_search(pruned_estimators);

    print_weights(weights_, "Cleaver Weights POST LS post-pruning");
  }

  // Put the new weights inside the ltr algorithm (including the pruned trees)
  if (update_model_) {
    bool res = algo->update_weights(weights_);
    if (!res)
      std::exit(EXIT_FAILURE);
  }

  score(training_dataset.get(), &training_score[0]);
  metric_on_training_ = metric->evaluate_dataset(training_dataset,
                                                     &training_score[0]);

  // compute the new ensemble size. The first line could be approx due to
  // the fact that line search could have set to 0 also ensemble not to prune
  // (as a result of the optimization process on the weights).
  unsigned int new_estimators_size = num_features - estimators_to_prune_;
  if (update_model_)
    new_estimators_size = algo->get_weights().size();

  std::cout << "# Model after optimization:" << std::endl;
  std::cout << std::fixed << std::setprecision(4);
  std::cout << "# --------------------------" << std::endl;
  std::cout << "#  size training validation" << std::endl;
  std::cout << "# --------------------------" << std::endl;
  std::cout << std::setw(7) << new_estimators_size;
  std::cout << std::setw(8) << metric_on_training_;
  if (validation_dataset) {
    std::vector<Score> validation_score(validation_dataset->num_instances());
    score(validation_dataset.get(), &validation_score[0]);
    metric_on_validation_ = metric->evaluate_dataset(
        validation_dataset, &validation_score[0]);
    std::cout << std::setw(10) << metric_on_validation_ << std::endl;
  }

  auto end = std::chrono::steady_clock::now();
  std::chrono::duration<double> elapsed = std::chrono::duration_cast<
      std::chrono::duration<double>>(end - begin);
  std::cout << std::endl;
  std::cout << "# \t Total training time: " << std::setprecision(2) <<
            elapsed.count() << " seconds" << std::endl;

  // Reset last_estimators_to_optimize_ to 0 in case of global optimization
  if (!opt_last_only)
    last_estimators_to_optimize_ = 0;
}

void Cleaver::score(data::Dataset *dataset, Score *scores) const {

  Feature *features = dataset->at(0, 0);
#pragma omp parallel for
  for (unsigned int s = 0; s < dataset->num_instances(); ++s) {
    size_t offset_feature = s * dataset->num_features();
    scores[s] = 0;
    // compute partialScore * weight for all the trees
    for (unsigned int f = 0; f < dataset->num_features(); ++f) {
      scores[s] += weights_[f] * features[offset_feature + f];
    }
  }
}

void Cleaver::import_weights_from_line_search(
    std::set<unsigned int> &pruned_estimators) {

  auto ls_weights = lineSearch_->get_weights();

  unsigned int ls_f = 0;
  for (unsigned int f = 0; f < weights_.size(); f++) {
    if (!pruned_estimators.count(f)) // skip pruned estimators
      weights_[f] = ls_weights[ls_f++];
  }

  assert(ls_f == ls_weights.size());
}

std::shared_ptr<data::Dataset> Cleaver::filter_dataset(
    std::shared_ptr<data::Dataset> dataset,
    std::set<unsigned int> &pruned_estimators) const {

  size_t estimators_to_select = dataset->num_features() - estimators_to_prune_;

  data::Dataset *filt_dataset = new data::Dataset(dataset->num_instances(),
                                                  estimators_to_select);

  // allocate feature vector
  std::vector<Feature> featureSelected(estimators_to_select);
  unsigned int skipped;

  for (unsigned int q = 0; q < dataset->num_queries(); q++) {
    std::shared_ptr<data::QueryResults> results = dataset->getQueryResults(q);
    const Feature *features = results->features();
    const Label *labels = results->labels();

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

bool Cleaver::update_weights(std::vector<double>& weights) {

  if (weights.size() != weights_.size()) {

    // copy the new weight vector, throwing away the old one (implicitly)
    weights_ = std::vector<double>(weights);

  } else {

    for (size_t k = 0; k < weights.size(); k++) {
      weights_[k] = weights[k];
    }
  }

  return true;
}

}  // namespace cleaver
}  // namespace post_learning
}  // namespace optimization
}  // namespace quickrank
