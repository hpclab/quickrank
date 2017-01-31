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
 * Contributor:
 *   HPC. Laboratory - ISTI - CNR - http://hpc.isti.cnr.it/
 */
#include "learning/forests/dart.h"

#include <fstream>
#include <iomanip>
#include <chrono>
#include <numeric>
#include <random>

#include "utils/radix.h"

namespace quickrank {
namespace learning {
namespace forests {

const std::string Dart::NAME_ = "DART";

const std::vector<std::string> Dart::samplingTypesNames = {
    "UNIFORM" //, "WEIGHTED"
};

const std::vector<std::string> Dart::normalizationTypesNames = {
    "TREE", "NONE", "WEIGHTED" //, "FOREST",
};

Dart::Dart(const pugi::xml_document &model) : LambdaMart(model) {

  sample_type = get_sampling_type(model.child("ranker").child("info")
      .child("sample_type").text().as_string());
  normalize_type = get_normalization_type(model.child("ranker").child("info")
      .child("normalize_type").text().as_string());
  rate_drop = model.child("ranker").child("info")
      .child("rate_drop").text().as_double();
  skip_drop = model.child("ranker").child("info")
      .child("skip_drop").text().as_double();
}

Dart::~Dart() {
  // TODO: fix the destructor...
}

std::ostream &Dart::put(std::ostream &os) const {

  os << "# Ranker: " << name() << std::endl
     << "# max no. of trees = " << ntrees_ << std::endl
     << "# no. of tree leaves = " << nleaves_ << std::endl
     << "# shrinkage = " << shrinkage_ << std::endl
     << "# min leaf support = " << minleafsupport_ << std::endl;
  if (nthresholds_)
    os << "# no. of thresholds = " << nthresholds_ << std::endl;
  else
    os << "# no. of thresholds = unlimited" << std::endl;
  if (valid_iterations_)
    os << "# no. of no gain rounds before early stop = " << valid_iterations_
       << std::endl;
  os << "# sample type = " << get_sampling_type(sample_type) << std::endl;
  os << "# normalization type = " << get_normalization_type(normalize_type)
     << std::endl;
  os << "# rate drop = " << rate_drop << std::endl;
  os << "# skip drop = " << skip_drop << std::endl;
  return os;
}

void Dart::learn(std::shared_ptr<quickrank::data::Dataset> training_dataset,
                 std::shared_ptr<quickrank::data::Dataset> validation_dataset,
                 std::shared_ptr<quickrank::metric::ir::Metric> scorer,
                 size_t partial_save, const std::string output_basename) {
  // ---------- Initialization ----------
  std::cout << "# Initialization";
  std::cout.flush();

  // to have the same behaviour
  std::srand(0);

  std::chrono::high_resolution_clock::time_point chrono_init_start =
      std::chrono::high_resolution_clock::now();

  // create a copy of the training datasets and put it in vertical format
  std::shared_ptr<quickrank::data::VerticalDataset> vertical_training(
      new quickrank::data::VerticalDataset(training_dataset));

  best_metric_on_validation_ = std::numeric_limits<double>::lowest();
  best_metric_on_training_ = std::numeric_limits<double>::lowest();
  best_model_ = 0;

  ensemble_model_.set_capacity(ntrees_);

  init(vertical_training);
  memset(scores_on_training_, 0, vertical_training->num_instances());

  if (validation_dataset) {
    scores_on_validation_ = new Score[validation_dataset->num_instances()]();
    memset(scores_on_validation_, 0, validation_dataset->num_instances());
  }

  // if the ensemble size is greater than zero, it means the learn method has
  // to start not from scratch but from a previously saved (intermediate) model
  if (ensemble_model_.is_notempty()) {
    best_model_ = ensemble_model_.get_size() - 1;

    // Update the model's outputs on all training samples
    score_dataset(training_dataset, scores_on_training_);
    // run metric
    best_metric_on_training_ = scorer->evaluate_dataset(
        vertical_training, scores_on_training_);

    if (validation_dataset) {
      // Update the model's outputs on all validation samples
      score_dataset(validation_dataset, scores_on_validation_);
      // run metric
      best_metric_on_validation_ = scorer->evaluate_dataset(
          validation_dataset, scores_on_validation_);
    }
  }

  auto chrono_init_end = std::chrono::high_resolution_clock::now();
  double init_time = std::chrono::duration_cast<std::chrono::duration<double>>(
      chrono_init_end - chrono_init_start).count();
  std::cout << ": " << std::setprecision(2) << init_time << " s." << std::endl;

  // ---------- Training ----------
  std::cout << std::fixed << std::setprecision(4);

  std::cout << "# Training:" << std::endl;
  std::cout << "# -------------------------" << std::endl;
  std::cout << "# iter. training validation" << std::endl;
  std::cout << "# -------------------------" << std::endl;

  // shows the performance of the already trained model..
  if (ensemble_model_.is_notempty()) {
    std::cout << std::setw(7) << ensemble_model_.get_size()
              << std::setw(9) << best_metric_on_training_;

    if (validation_dataset)
      std::cout << std::setw(9) << best_metric_on_validation_;

    std::cout << " *" << std::endl;
  }

  auto chrono_train_start = std::chrono::high_resolution_clock::now();

  quickrank::MetricScore metric_on_training =
      std::numeric_limits<double>::lowest();
  quickrank::MetricScore metric_on_validation =
      std::numeric_limits<double>::lowest();

  // start iterations from 0 or (ensemble_size - 1)
  for (size_t m = ensemble_model_.get_size(); m < ntrees_; ++m) {
    if (validation_dataset
        && (valid_iterations_ && m > best_model_ + valid_iterations_))
      break;

    std::vector<double> weights = ensemble_model_.get_weights();

    double prob_skip_dropout = (double)rand() / (double)(RAND_MAX);
    int trees_to_dropout = (int) round(rate_drop * weights.size());
    double metric_on_training_dropout;
    double metric_on_validation_dropout;
    std::vector<int> dropped_trees;
    bool dropout_better_than_full = false;
    if (trees_to_dropout > 0 && prob_skip_dropout > skip_drop) {

      dropped_trees = select_trees_to_dropout(weights, trees_to_dropout);

      std::vector<double> dropped_weights(weights);
      for (auto idx: dropped_trees) {
        dropped_weights[idx] = 0;
      }

      ensemble_model_.update_ensemble_weights(dropped_weights, false);

      // Update the model's outputs on all training samples
      score_dataset(training_dataset, scores_on_training_);
//      update_modelscores(training_dataset, false,
//                         scores_on_training_, dropped_trees);

      // run metric
      metric_on_training_dropout = scorer->evaluate_dataset(
          training_dataset, scores_on_training_);

      if (validation_dataset) {
        // Update the model's outputs on all validation samples
        score_dataset(validation_dataset, scores_on_validation_);
//        update_modelscores(validation_dataset, false,
//                           scores_on_validation_, dropped_trees);
        // run metric
        metric_on_validation_dropout = scorer->evaluate_dataset(
            validation_dataset, scores_on_validation_);
      }

      // Apply the removal of the dropped trees from the forest
//      ensemble_model_.update_ensemble_weights(dropped_weights, false);

      if (validation_dataset) {
        if (metric_on_validation_dropout > metric_on_validation)
          dropout_better_than_full = true;
      } else {
        if (metric_on_training_dropout > metric_on_training)
          dropout_better_than_full = true;
      }

    } else {
      trees_to_dropout = 0;
    }

    compute_pseudoresponses(vertical_training, scorer.get());

    // update the histogram with these training_setting labels
    // (the feature histogram will be used to find the best tree rtnode)
    hist_->update(pseudoresponses_, vertical_training->num_instances());

    //Fit a regression tree
    std::unique_ptr<RegressionTree> tree =
        fit_regressor_on_gradient(vertical_training);

    // add this tree to the ensemble (our model)
    ensemble_model_.push(tree->get_proot(), shrinkage_, 0);  // maxlabel);

    // ----------
    double metric_on_training_fit;
    double metric_on_validation_fit;
    // Update the model's outputs on all training samples
    score_dataset(training_dataset, scores_on_training_);
//      update_modelscores(training_dataset, false,
//                         scores_on_training_, dropped_trees);

    // run metric
    metric_on_training_fit = scorer->evaluate_dataset(
        training_dataset, scores_on_training_);

    if (validation_dataset) {
      // Update the model's outputs on all validation samples
      score_dataset(validation_dataset, scores_on_validation_);
//        update_modelscores(validation_dataset, false,
//                           scores_on_validation_, dropped_trees);
      // run metric
      metric_on_validation_fit = scorer->evaluate_dataset(
          validation_dataset, scores_on_validation_);
    }

    bool fit_after_dropout_better_than_full = false;
    if (validation_dataset) {
      if (metric_on_validation_fit > metric_on_validation)
        fit_after_dropout_better_than_full = true;
    } else {
      if (metric_on_training_fit > metric_on_training)
        fit_after_dropout_better_than_full = true;
    }
    // --------------------

    if (dropped_trees.size() > 0) {
      // Normalize the weight vector increased by the last added tree
      weights.push_back(shrinkage_);
      normalize_trees(weights, dropped_trees);
      ensemble_model_.update_ensemble_weights(weights, false);
    }

    // add the last tree for the update of the modelscores
    dropped_trees.push_back(ensemble_model_.get_size()-1);

    // Update the model's outputs on all training samples
//    update_modelscores(vertical_training, true,
//                       scores_on_training_, dropped_trees);
    score_dataset(training_dataset, scores_on_training_);

    // run metric
    metric_on_training = scorer->evaluate_dataset(vertical_training,
                                                  scores_on_training_);

    //show results
    std::cout << std::setw(7) << m + 1 << std::setw(9) << metric_on_training;

    //Evaluate the current model on the validation data (if available)
    if (validation_dataset) {

//      update_modelscores(validation_dataset, true,
//                         scores_on_validation_, dropped_trees);
      score_dataset(validation_dataset, scores_on_validation_);

      // run metric
      metric_on_validation = scorer->evaluate_dataset(
          validation_dataset, scores_on_validation_);
      std::cout << std::setw(9) << metric_on_validation;

      if (metric_on_validation > best_metric_on_validation_) {
        best_metric_on_training_ = metric_on_training;
        best_metric_on_validation_ = metric_on_validation;
        best_model_ = ensemble_model_.get_size() - 1;
        std::cout << " *";
      }
    } else {
      if (metric_on_training > best_metric_on_training_) {
        best_metric_on_training_ = metric_on_training;
        best_model_ = ensemble_model_.get_size() - 1;
        std::cout << " *";
      }
    }

    std::string betterDrop = "  ";
    std::string betterFit = "  ";
    if (dropout_better_than_full)
      betterDrop = " *";
    if (fit_after_dropout_better_than_full)
      betterFit = " *";

    std::cout << "\t[ " << metric_on_training_dropout << " -> "
              << metric_on_training_fit << " -> "
              << metric_on_training << " | "
              << metric_on_validation_dropout << betterDrop << " -> "
              << metric_on_validation_fit << betterFit << " -> "
              << metric_on_validation;
    std::cout << " }";

    std::cout << "\t" << trees_to_dropout << " \t Dropped Trees: [ ";
    for (unsigned int i=0; i<dropped_trees.size()-1; ++i)
      std::cout << dropped_trees[i] << " ";
    std::cout << "]" << std::endl;

    if (partial_save != 0 and !output_basename.empty()
        and (m + 1) % partial_save == 0) {
      save(output_basename, m + 1);
    }
  }

  //Rollback to the best model observed on the validation data
  if (validation_dataset) {
    while (ensemble_model_.is_notempty()
        && ensemble_model_.get_size() > best_model_ + 1) {
      ensemble_model_.pop();
    }
  }

  auto chrono_train_end = std::chrono::high_resolution_clock::now();
  double train_time = std::chrono::duration_cast<std::chrono::duration<double>>(
      chrono_train_end - chrono_train_start).count();

  //Finishing up
  std::cout << std::endl;
  std::cout << *scorer << " on training data = " << best_metric_on_training_
            << std::endl;

  if (validation_dataset) {
    std::cout << *scorer << " on validation data = "
              << best_metric_on_validation_ << std::endl;
  }

  clear(vertical_training->num_features());

  std::cout << std::endl;
  std::cout << "#\t Training Time: " << std::setprecision(2) << train_time
            << " s." << std::endl;
}

pugi::xml_document *Dart::get_xml_model() const {

  pugi::xml_document *doc = new pugi::xml_document();
  pugi::xml_node root = doc->append_child("ranker");
  pugi::xml_node info = root.append_child("info");

  info.append_child("type").text() = name().c_str();
  info.append_child("trees").text() = ntrees_;
  info.append_child("leaves").text() = nleaves_;
  info.append_child("shrinkage").text() = shrinkage_;
  info.append_child("leafsupport").text() = minleafsupport_;
  info.append_child("discretization").text() = nthresholds_;
  info.append_child("estop").text() = valid_iterations_;

  info.append_child("sample_type").text() =
      get_sampling_type(sample_type).c_str();
  info.append_child("normalize_type").text() =
      get_normalization_type(normalize_type).c_str();
  info.append_child("rate_drop").text() = rate_drop;
  info.append_child("skip_drop").text() = skip_drop;

  ensemble_model_.append_xml_model(root);

  return doc;
}

bool Dart::import_model_state(LTR_Algorithm &other) {

  // Check the object is derived from Dart
  try
  {
    Dart& otherCast = dynamic_cast<Dart&>(other);

    if (std::abs(shrinkage_ - otherCast.shrinkage_) > 0.000001 ||
        nthresholds_ != otherCast.nthresholds_ ||
        nleaves_ != otherCast.nleaves_ ||
        minleafsupport_ != otherCast.minleafsupport_ ||
        valid_iterations_ != otherCast.valid_iterations_ ||
        sample_type != otherCast.sample_type ||
        normalize_type != otherCast.normalize_type ||
        rate_drop != otherCast.rate_drop ||
        skip_drop != otherCast.skip_drop)
      return false;

    // Move assignment operator
    // Move the ownership of the ensemble object to the current model
    ensemble_model_ = std::move(otherCast.ensemble_model_);
  }
  catch(std::bad_cast)
  {
    return false;
  }

  return true;
}

void Dart::update_modelscores(std::shared_ptr<data::Dataset> dataset,
                              bool add, Score *scores,
                              std::vector<int>& trees_to_update) {

  const quickrank::Feature *d = dataset->at(0, 0);
  const size_t offset = 1;
  const size_t num_features = dataset->num_features();
  const double sign = add ? 1.0f : -1.0f;

  for (int t: trees_to_update) {
    #pragma omp parallel for
    for (size_t i = 0; i < dataset->num_instances(); ++i) {
      scores[i] += sign * ensemble_model_.getWeight(t) *
          ensemble_model_.getTree(t)->score_instance(d + i * num_features, offset);
    }
  }
}

void Dart::update_modelscores(std::shared_ptr<data::VerticalDataset> dataset,
                              bool add, Score *scores,
                              std::vector<int>& trees_to_update) {

  const quickrank::Feature *d = dataset->at(0, 0);
  const size_t offset = dataset->num_instances();
  const double sign = add ? 1.0f : -1.0f;
  for (int t: trees_to_update) {
    #pragma omp parallel for
    for (size_t i = 0; i < dataset->num_instances(); ++i) {
      scores[i] += sign * ensemble_model_.getWeight(t) *
          ensemble_model_.getTree(t)->score_instance(d + i, offset);
    }
  }
}

std::vector<int> Dart::select_trees_to_dropout(std::vector<double>& weights,
                                               int trees_to_dropout) {

  if (trees_to_dropout == 0)
    return std::vector<int>(0);

  std::vector<int> dropped;

  if (sample_type == SamplingType::UNIFORM) {

    std::vector<int> idx(weights.size());
    std::iota(idx.begin(), idx.end(), 0);

    struct RNG {
      int operator() (int n) {
        return std::rand() / (1.0 + RAND_MAX) * n;
      }
    };

    // Permute idx
    std::random_shuffle(idx.begin(), idx.end(), RNG());

    for(int i=0; i<trees_to_dropout; ++i)
      dropped.push_back(idx[i]);

  }
//  else if (sample_type == SamplingType::WEIGHTED) {
//
//    // TODO: implement weighted sampling
//  }

  return dropped;
}

void Dart::normalize_trees(std::vector<double>& weights,
                           std::vector<int> dropped_trees) {

  int k = dropped_trees.size();
  if (k == 0)
    return;

  if (normalize_type == NormalizationType::TREE) {

    // Normalize last added tree
    weights.back() *= (1.0f / k);

    // Normalize dropped trees and last added tree
    double norm = (double) k / (k + 1);

    weights.back() *= norm;
    for (int idx: dropped_trees) {
      weights[idx] *= norm;
    }

  } else if (normalize_type == NormalizationType::NONE) {

    // nothing to do

  } else if (sample_type == SamplingType::WEIGHTED) {

    // Sum of the weights of the dropped trees + last added tree
//    double sum = weights[weights.size() - 1];
    double sum = 0;
    for (int t: dropped_trees)
      sum += weights[t];

    weights.back() /= sum;

    double sumWithLast = sum + weights[weights.size() - 1];
    double norm = (double) sum / sumWithLast;
    weights.back() *= norm;
    for (int t: dropped_trees)
      weights[t] *= norm;
  }

//  else if (normalize_type == NormalizationType::FOREST) {
//
//    // TODO: implement forest normalization
//
//  }

}

}  // namespace forests
}  // namespace learning
}  // namespace quickrank
