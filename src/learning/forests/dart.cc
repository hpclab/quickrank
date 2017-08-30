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

#include <fstream>
#include <iomanip>
#include <chrono>
#include <numeric>
#include <random>
#include <assert.h>

#include "learning/forests/dart.h"
#include "utils/radix.h"

namespace quickrank {
namespace learning {
namespace forests {

const std::string Dart::NAME_ = "DART";

const std::vector<std::string> Dart::samplingTypesNames = {
    "UNIFORM" , "WEIGHTED", "WEIGHTED_INV", "TOP_FIFTY", "CONTR", "CONTR_INV",
    "WCONTR", "WCONTR_INV", "TOP_WCONTR", "LESS_WCONTR"
};

const std::vector<std::string> Dart::normalizationTypesNames = {
    "TREE", "NONE", "WEIGHTED", "FOREST", "TREE_ADAPTIVE", "LINESEARCH",
    "TREE_BOOST3", "CONTR", "WCONTR", "LMART_ADAPTIVE"
};

const std::vector<std::string> Dart::adaptiveTypeNames = {
    "FIXED", "PLUS1_DIV2", "PLUSHALF_DIV2", "PLUSONETHIRD_DIV2",
    "PLUSHALF_RESET", "PLUSHALF_RESET_LB1_UB5", "PLUSHALF_RESET_LB1_UB10",
    "PLUSHALF_RESET_LB1_UBRD"
};

Dart::Dart(const pugi::xml_document &model) : LambdaMart(model) {

  sample_type = get_sampling_type(model.child("ranker").child("info")
      .child("sample_type").text().as_string());
  normalize_type = get_normalization_type(model.child("ranker").child("info")
      .child("normalize_type").text().as_string());

  if (model.child("ranker").child("info").child("adaptive_type")) {
    adaptive_type = get_adaptive_type(model.child("ranker").child("info")
                                .child("adaptive_type").text().as_string());
  } else
    adaptive_type = AdaptiveType::FIXED;

  rate_drop = model.child("ranker").child("info")
      .child("rate_drop").text().as_double();
  skip_drop = model.child("ranker").child("info")
      .child("skip_drop").text().as_double();
  keep_drop = model.child("ranker").child("info")
      .child("keep_drop").text().as_bool();

  if (model.child("ranker").child("info").child("best_on_train")) {
    best_on_train = model.child("ranker").child("info")
        .child("best_on_train").text().as_bool();
  } else
    best_on_train = false;

  if (model.child("ranker").child("info").child("random_keep")) {
    random_keep = model.child("ranker").child("info")
        .child("random_keep").text().as_double();
  } else
    random_keep = 0;

  if (model.child("ranker").child("info").child("drop_on_best")) {
    best_on_train = model.child("ranker").child("info")
        .child("drop_on_best").text().as_bool();
  } else
    drop_on_best = false;
}

Dart::~Dart() {
  // TODO: fix the destructor...
}

void
Dart::init(std::shared_ptr<quickrank::data::VerticalDataset> training_dataset) {
  LambdaMart::init(training_dataset);
  const size_t nentries = training_dataset->num_instances();
  scores_contribution_ = new double[nentries]();  //0.0f initialized
}

void Dart::clear(size_t num_features) {
  LambdaMart::clear(num_features);
  if (scores_contribution_)
    delete[] scores_contribution_;

  // Reset pointers to internal data structures
  scores_contribution_ = NULL;
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
  info.append_child("adaptive_type").text() =
      get_adaptive_type(adaptive_type).c_str();
  info.append_child("rate_drop").text() = rate_drop;
  info.append_child("skip_drop").text() = skip_drop;
  info.append_child("best_on_train").text() = best_on_train;
  info.append_child("random_keep").text() = random_keep;
  info.append_child("drop_on_best").text() = drop_on_best;

  ensemble_model_.append_xml_model(root);

  return doc;
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
  os << "# adaptive type = " << get_adaptive_type(adaptive_type) << std::endl;
  os << "# rate drop = " << rate_drop << std::endl;
  os << "# skip drop = " << skip_drop << std::endl;
  os << "# keep drop = " << keep_drop << std::endl;
  os << "# best on train = " << best_on_train << std::endl;
  os << "# keep dropout at random = " << random_keep << std::endl;
  os << "# keep dropout based on best = " << drop_on_best << std::endl;
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
  size_t best_iter_ = 0;
  std::vector<double> best_weights;

  ensemble_model_.set_capacity(ntrees_ + valid_iterations_);

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
    best_iter_ = best_model_;
    best_weights = ensemble_model_.get_weights();

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

  size_t dropped_before_cleaning = 0;
  // start iterations from 0 or (ensemble_size - 1)
  size_t m = -1;
  size_t last_iteration_global_scoring = 0;
  std::vector<int> counts;
  std::vector<double> performance_on_validation;
  std::vector<double> dropout_factor_per_iter;
  while ((ensemble_model_.get_size() - dropped_before_cleaning) < ntrees_) {
    ++m;
//  for (size_t m = ensemble_model_.get_size(); m < ntrees_; ++m) {
    if (validation_dataset
        && (valid_iterations_ && m > best_iter_ + valid_iterations_))
      break;

    std::vector<double> orig_weights = ensemble_model_.get_weights();

    int trees_to_dropout = get_number_of_trees_to_dropout(
        performance_on_validation, dropout_factor_per_iter,
        best_metric_on_validation_,
        dropped_before_cleaning);

    double prob_random_keep = (double) rand() / (double) (RAND_MAX);
    bool random_keep_iter = trees_to_dropout > 0 &&
        prob_random_keep <= random_keep;

    double metric_on_training_dropout = 0;
    double metric_on_validation_dropout = 0;
    std::vector<int> dropped_trees;
    bool dropout_better_than_full = false;

    std::vector<double> dropped_weights(orig_weights);
    if (trees_to_dropout > 0) {

      dropped_trees = select_trees_to_dropout(orig_weights, trees_to_dropout);

      // Subtracts from overall scores the dropped trees (for train and vali)
      update_modelscores(training_dataset, false,
                         scores_on_training_, dropped_trees);
      metric_on_training_dropout = scorer->evaluate_dataset(
          training_dataset, scores_on_training_);

      if (validation_dataset) {
        update_modelscores(validation_dataset, false,
                           scores_on_validation_, dropped_trees);
        metric_on_validation_dropout = scorer->evaluate_dataset(
            validation_dataset, scores_on_validation_);
      }

      if (validation_dataset) {
        if (metric_on_validation_dropout > metric_on_validation)
          dropout_better_than_full = true;
      } else {
        if (metric_on_training_dropout > metric_on_training)
          dropout_better_than_full = true;
      }

//      std::vector<double> dropped_weights(weights);
      for (auto idx: dropped_trees) {
        dropped_weights[idx] = 0;
      }
      ensemble_model_.update_ensemble_weights(dropped_weights, false);
    }

    compute_pseudoresponses(vertical_training, scorer.get());

    // update the histogram with these training_setting labels
    // (the feature histogram will be used to find the best tree rtnode)
    hist_->update(pseudoresponses_, vertical_training->num_instances());

    // Fit a regression tree
    std::shared_ptr<RegressionTree> tree =
        fit_regressor_on_gradient(vertical_training);

    // Update scores_contribution_ including last tree
    update_contribution_scores(training_dataset, tree,
                               ensemble_model_.get_size());

    // Calculate the weight of the new tree
    double tree_weight = get_weight_last_tree(training_dataset,
                                              scorer,
                                              dropped_weights,
                                              dropped_trees,
                                              tree);

    // add this tree to the ensemble (our model)
    ensemble_model_.push(tree->get_proot(), tree_weight, 0);

    // Init the counter of the last added tree
    counts.push_back(0);

    double metric_on_training_fit = std::numeric_limits<double>::lowest();
    double metric_on_validation_fit = std::numeric_limits<double>::lowest();

    int lastTreeIndex = (int) ensemble_model_.get_size() - 1;
    std::vector<int> lastTree = {lastTreeIndex};
    update_modelscores(training_dataset, true,
                       scores_on_training_, lastTree);
    metric_on_training_fit = scorer->evaluate_dataset(training_dataset,
                                                      scores_on_training_);

    if (validation_dataset) {
        update_modelscores(validation_dataset, true,
                           scores_on_validation_, lastTree);
      metric_on_validation_fit = scorer->evaluate_dataset(validation_dataset,
                                                          scores_on_validation_);
    }

    bool fit_after_dropout_improvement = false;
    if (trees_to_dropout > 0) {

      double reference_metric_training = metric_on_training;
      double reference_metric_validation = metric_on_validation;
      if (drop_on_best) {
        reference_metric_training = best_metric_on_training_;
        reference_metric_validation = best_metric_on_validation_;
      }

      if (validation_dataset) {
        if (metric_on_validation_fit > reference_metric_validation)
          fit_after_dropout_improvement = true;
      } else {
        if (metric_on_training_fit > reference_metric_training)
          fit_after_dropout_improvement = true;
      }
    }

    if (keep_drop && (fit_after_dropout_improvement || random_keep_iter)) {

      dropped_before_cleaning += trees_to_dropout;

      metric_on_training = metric_on_training_fit;
      metric_on_validation = metric_on_validation_fit;

    } else {

      // Reset the original scores before doing normalization
      update_modelscores(training_dataset, false,
                         scores_on_training_, lastTree);
      if (validation_dataset) {
        update_modelscores(validation_dataset, false,
                           scores_on_validation_, lastTree);
      }

      if (trees_to_dropout > 0) {
        // Normalize the weight vector and add the last tree
        normalize_trees_restore_drop(orig_weights, dropped_trees, tree_weight);
        ensemble_model_.update_ensemble_weights(orig_weights, false);
      }

      // add the last tree for the update of the modelscores
      dropped_trees.push_back( (int) ensemble_model_.get_size() - 1);

      update_modelscores(training_dataset, true,
                         scores_on_training_, dropped_trees);
      metric_on_training = scorer->evaluate_dataset(vertical_training,
                                                    scores_on_training_);

      if (validation_dataset) {
        update_modelscores(validation_dataset, true,
                           scores_on_validation_, dropped_trees);

        metric_on_validation = scorer->evaluate_dataset(validation_dataset,
                                                        scores_on_validation_);
      }
    }

    // Find trees with count above the threshold
    std::vector<int> trees_to_drop_by_count;

    //show results
    std::cout << std::setw(7) << m + 1 << std::setw(9) << metric_on_training;

    bool best_improved = false;
    if (validation_dataset && !best_on_train) {

      // run metric
      std::cout << std::setw(9) << metric_on_validation;

      if (metric_on_validation > best_metric_on_validation_)
        best_improved = true;

    } else {

      if (metric_on_training > best_metric_on_training_)
        best_improved = true;
    }

    bool best_vali_improved = false;
    if (validation_dataset && best_on_train &&
        metric_on_validation > best_metric_on_validation_) {
      best_vali_improved = true;
      best_metric_on_validation_ = metric_on_validation;
    }

    if (best_improved) {

      best_metric_on_training_ = metric_on_training;
      if (!best_on_train)
        best_metric_on_validation_ = metric_on_validation;
      best_iter_ = m;
      std::cout << " *";

      // Removes trees with 0-weight from the ensemble
      ensemble_model_.filter_out_zero_weighted_trees();
      // Removes also the contributions from the scores array
      filter_out_zero_weighted_contributions(ensemble_model_.get_weights());

      // Update the best weights vector with remaining trees
      best_weights = ensemble_model_.get_weights();
      best_model_ = ensemble_model_.get_size();
      dropped_before_cleaning = 0;
    }

    std::string improved = " ";
    if (best_vali_improved)
      improved += "*";
    else
      improved += " ";

    std::string betterDrop = "  ";
    std::string betterFit = "  ";
    if (dropout_better_than_full)
      betterDrop = " *";
    if (fit_after_dropout_improvement)
      betterFit = " *";

    std::cout << "\t[ " << metric_on_training_dropout << " - "
              << metric_on_training_fit << " - "
              << metric_on_training << " | "
              << metric_on_validation_dropout << betterDrop << " - "
              << metric_on_validation_fit << betterFit << " - "
              << metric_on_validation << improved;
    std::cout << "]";

    std::cout << " \t" << trees_to_dropout << " Dropped Trees "
              << "- Ensemble size: "
              << ensemble_model_.get_size() - dropped_before_cleaning;
    if (keep_drop && fit_after_dropout_improvement)
        std::cout << " - Keep Dropout";
    else if (random_keep_iter)
      std::cout << " - Keep Dropout (RANDOM)";
    else if (trees_to_dropout > 0)
      std::cout << " - Dropout";
    if (trees_to_drop_by_count.size() > 0)
      std::cout << " - Count Drop: " << trees_to_drop_by_count.size();

    if (best_improved) {
      std::cout << " - CLEANED";
      if ( (m - last_iteration_global_scoring) > 10) {
        score_dataset(training_dataset, scores_on_training_);
        if (validation_dataset)
          score_dataset(validation_dataset, scores_on_validation_);
        std::cout << " (update)";
        last_iteration_global_scoring = m;
      }
    }

    std::cout << std::endl;

    performance_on_validation.push_back(metric_on_validation);

    if (partial_save != 0 and !output_basename.empty()
        and (ensemble_model_.get_size() - dropped_before_cleaning) % partial_save == 0) {
      save(output_basename, ensemble_model_.get_size() - dropped_before_cleaning);
    }
  }

  //Rollback to the best model observed on the validation data
  if (validation_dataset) {
    while (ensemble_model_.is_notempty()
        && ensemble_model_.get_size() > best_model_) {
      ensemble_model_.pop();
    }
    ensemble_model_.update_ensemble_weights(best_weights, true);
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
  const double sign = add ? 1.0 : -1.0;

  for (int t: trees_to_update) {
    #pragma omp parallel for
    for (size_t i = 0; i < dataset->num_instances(); ++i) {
      scores[i] += sign * ensemble_model_.getWeight(t) *
          ensemble_model_.getTree(t)->score_instance(d + i * num_features, offset);
    }
  }
}

//void Dart::update_modelscores(std::shared_ptr<data::Dataset> dataset,
//                              bool add, Score *scores,
//                              std::vector<int>& trees_to_update) {
//
//  std::vector<double> orig_weights = ensemble_model_.get_weights();
//
//  if (!add) {
//    std::vector<double> new_weights(orig_weights);
//    for (int idx: trees_to_update)
//      new_weights[idx] = 0;
//
//    ensemble_model_.update_ensemble_weights(new_weights, false);
//    score_dataset(dataset, scores);
//    ensemble_model_.update_ensemble_weights(orig_weights, false);
//
//  } else {
//
//    score_dataset(dataset, scores);
//  }
//}

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

void Dart::update_contribution_scores(std::shared_ptr<data::Dataset> dataset,
                                      std::shared_ptr<RegressionTree> tree,
                                      int new_index) {

  const quickrank::Feature *d = dataset->at(0, 0);
  const size_t offset = 1;
  const size_t num_features = dataset->num_features();
  const size_t num_instances = dataset->num_instances();

  double contribution = 0;
  RTNode* root = tree->get_proot();
  #pragma omp parallel for reduction(+:contribution)
  for (size_t i = 0; i < dataset->num_instances(); ++i) {
    contribution += fabs(root->score_instance(d + i * num_features, offset));
  }

  scores_contribution_[new_index] = contribution / num_instances;
}

std::vector<int> Dart::select_trees_to_dropout(std::vector<double>& weights,
                                               size_t trees_to_dropout) {

  if (trees_to_dropout == 0)
    return std::vector<int>(0);

  std::vector<int> dropped;

  if (sample_type == SamplingType::UNIFORM ||
      sample_type == SamplingType::TOP_FIFTY) {

    size_t size = weights.size();
    if (sample_type == SamplingType::TOP_FIFTY)
      size = (size_t) round(size / 2);
    std::vector<int> idx(size);
    std::iota(idx.begin(), idx.end(), 0);

    struct RNG {
      int operator() (int n) {
        return (int) (std::rand() / (1.0 + RAND_MAX) * n);
      }
    };

    // Permute idx
    std::random_shuffle(idx.begin(), idx.end(), RNG());

    for(size_t i=0; dropped.size() < trees_to_dropout && i < idx.size(); ++i)
      if (weights[idx[i]] > 0)
        dropped.push_back(idx[i]);

  } else if ( sample_type == SamplingType::WEIGHTED ||
              sample_type == SamplingType::WEIGHTED_INV) {

    double sumWeights = 0;
    for (auto w: weights)
      sumWeights += w;

    std::vector<double> prob(weights);
    std::vector<double> cumProb(weights.size());

    for(int i=0; dropped.size() < trees_to_dropout; ++i) {

      // Simulate the generation of a random permutation with
      // different probability for each element to be selected
      for (unsigned int i=0; i<weights.size(); ++i) {
        if (prob[i] != 0)
          prob[i] = weights[i] / sumWeights;
        if (sample_type == SamplingType::WEIGHTED_INV)
          prob[i] = 1 - prob[i];
        cumProb[i] = prob[i];
        if (i>0)
          cumProb[i] += cumProb[i-1];
      }

      double select = (double) rand() / (double) (RAND_MAX);

      int index = binary_search(cumProb, select);
      // We are trying to drop-out more than valid elements (!= 0)
      if (index == -1)
        break;

      dropped.push_back(index);
      sumWeights -= weights[index];
      prob[index] = 0;
    }

  } else if ( sample_type == SamplingType::CONTR ||
              sample_type == SamplingType::CONTR_INV ||
              sample_type == SamplingType::WCONTR ||
              sample_type == SamplingType::WCONTR_INV) {

    double sumContributions = 0;
    for (size_t i = 0; i < weights.size(); ++i) {
      if (weights[i] == 0) // avoid considering the 0-weighted trees (removed)
        continue;

      double weight = 1;
      if (sample_type == SamplingType::WCONTR ||
          sample_type == SamplingType::WCONTR_INV)
        weight = weights[i];
      sumContributions += weight * scores_contribution_[i];
    }

    std::vector<double> prob(weights);
    std::vector<double> cumProb(weights.size());

    for(int i=0; dropped.size() < trees_to_dropout; ++i) {

      // Simulate the generation of a random permutation with
      // different probability for each element to be selected
      for (unsigned int i=0; i<weights.size(); ++i) {

        double weight = 1;
        if (sample_type == SamplingType::WCONTR ||
            sample_type == SamplingType::WCONTR_INV)
          weight = weights[i];

        if (weights[i] > 0)
          prob[i] = weight * scores_contribution_[i] / sumContributions;

        if (sample_type == SamplingType::CONTR_INV ||
            sample_type == SamplingType::WCONTR_INV)
          prob[i] = 1 - prob[i];
        cumProb[i] = prob[i];
        if (i > 0)
          cumProb[i] += cumProb[i-1];
      }

      double select = (double) rand() / (double) (RAND_MAX);

      int index = binary_search(cumProb, select);
      // We are trying to drop-out more than valid elements (!= 0)
      if (index == -1)
        break;

      dropped.push_back(index);
      double weight = 1;
      if (sample_type == SamplingType::WCONTR ||
          sample_type == SamplingType::WCONTR_INV)
        weight = weights[i];
      sumContributions -= weight * scores_contribution_[i];
      prob[index] = 0;
    }

  } else if ( sample_type == SamplingType::TOP_WCONTR ||
      sample_type == SamplingType::LESS_WCONTR) {

    std::vector<double> contr(weights);
    for (unsigned int i=0; i<weights.size(); ++i)
      contr[i] = weights[i] * scores_contribution_[i];

    for(int i=0; dropped.size() < trees_to_dropout; ++i) {

      std::vector<double>::const_iterator i_contr;
      if (sample_type == SamplingType::TOP_WCONTR)
        i_contr = std::max_element(contr.cbegin(), contr.cend());
      else
        i_contr = std::min_element(contr.cbegin(), contr.cend());
      int p = (int) std::distance(contr.cbegin(), i_contr);

      dropped.push_back(p);
    }

  }

  return dropped;
}

void Dart::normalize_trees_restore_drop(std::vector<double>& weights,
                                        std::vector<int> dropped_trees,
                                        double last_tree_weight) {

  // This function has to add the weight of the last trained tree
  // to the vector of weights

  size_t k = dropped_trees.size();

  if (normalize_type == NormalizationType::TREE ||
      normalize_type == NormalizationType::TREE_ADAPTIVE ||
      normalize_type == NormalizationType::TREE_BOOST3) {

    double alpha = 1;
    if (normalize_type == NormalizationType::TREE_BOOST3)
      alpha = 3;

    // Normalize last added tree
    weights.push_back( (shrinkage_ * alpha) / ( (shrinkage_ * alpha) + k) );

    // Normalize dropped trees and last added tree
    double norm = (double) k / (k + (shrinkage_ * alpha) );
    for (int idx: dropped_trees)
      weights[idx] *= norm;

  } else if (normalize_type == NormalizationType::NONE) {

    weights.push_back(shrinkage_);

  } else if (normalize_type == NormalizationType::WEIGHTED) {

    // Sum of the weights of the dropped trees + last added tree
//    double sum = weights[weights.size() - 1];
    double sum = 0;
    for (int t: dropped_trees)
      sum += weights[t];

    double sumWithLast = sum + shrinkage_;
    double norm = sum / sumWithLast;
    weights.push_back(shrinkage_ / sumWithLast);
    for (int t: dropped_trees)
      weights[t] *= norm;

  } else if (normalize_type == NormalizationType::FOREST) {

    weights.push_back(shrinkage_ / (1 + shrinkage_));

    double norm = 1 / (1 + shrinkage_);
    for (int idx: dropped_trees)
      weights[idx] *= norm;

  } else if (normalize_type == NormalizationType::LINESEARCH) {

    // Normalize last added tree
    weights.push_back(last_tree_weight / (last_tree_weight + k) );

    // Normalize dropped trees and last added tree
    double norm = (double) k / (k + last_tree_weight);
    for (int idx: dropped_trees)
      weights[idx] *= norm;

  } else if ( normalize_type == NormalizationType::CONTR ||
              normalize_type == NormalizationType::WCONTR) {

    double dropped_contribution = 0;
    double weight = 1;
    for (auto t: dropped_trees) {
      if (normalize_type == NormalizationType::WCONTR)
        weight = weights[t];
      dropped_contribution += weight * scores_contribution_[t];
    }

    // weights contains the weights excluding the last added tree
    // so, size is the index of the new tree
    weight = 1;
    if (normalize_type == NormalizationType::WCONTR)
      weight = last_tree_weight;
    double contribution_last_tree = weight *
                                    scores_contribution_[weights.size()];

    double sum_contribution = dropped_contribution + contribution_last_tree;
    double norm = dropped_contribution / sum_contribution;
    weights.push_back(contribution_last_tree / sum_contribution);
    for (int t: dropped_trees)
      weights[t] *= norm;
  }
}

double Dart::get_weight_last_tree(std::shared_ptr<data::Dataset> dataset,
                                  std::shared_ptr<metric::ir::Metric> scorer,
                                  std::vector<double> &weights,
                                  std::vector<int> dropped_trees,
                                  std::shared_ptr<RegressionTree> tree) {

  size_t k = dropped_trees.size();

  if (normalize_type == NormalizationType::TREE) {

    return shrinkage_;

  } else if (normalize_type == NormalizationType::NONE) {

    return shrinkage_;

  } else if (normalize_type == NormalizationType::WEIGHTED) {

    return shrinkage_;

  } else if (normalize_type == NormalizationType::FOREST) {

    return shrinkage_;

  } else if (normalize_type == NormalizationType::TREE_ADAPTIVE) {

    return shrinkage_ / (shrinkage_ + k);

  } else if (normalize_type == NormalizationType::TREE_BOOST3) {

    double alfa = 3;
    return (shrinkage_ * alfa) / ( (shrinkage_ * alfa) + k);

  } else if (normalize_type == NormalizationType::LINESEARCH) {

    // scores already contains the sum of scores per instance except the last
    // trained tree

    const quickrank::Feature *d = dataset->at(0, 0);
    const size_t offset = 1;
    const size_t num_features = dataset->num_features();
    const size_t num_instances = dataset->num_instances();

    const int num_points = 16;
    const double window_size = 1;
    const double starting_weight = 1.0f;
    const double step = 2 * window_size / num_points;

    std::vector<double> weights;
    weights.reserve(num_points + 1); // don't know the exact number of points
    for (double weight = starting_weight - window_size;
         weight <= starting_weight + window_size; weight += step) {
      if (weight > 0)
        weights.push_back(weight);
    }

    std::vector<Score> score_instance_last_tree(num_instances);
    #pragma omp parallel for
    for (size_t i = 0; i < dataset->num_instances(); ++i) {
      score_instance_last_tree[i] += tree->get_proot()
          ->score_instance(d + i * num_features, offset);
    }


    std::vector<Score> scores(num_instances * (weights.size()), 0.0);
    std::vector<MetricScore> metric_scores(weights.size(), 0.0);

    #pragma omp parallel for
    for (unsigned int p = 0; p < weights.size(); ++p) {
      for (unsigned int s = 0; s < num_instances; ++s) {
        // Scores without last tree + weight * score_last_tree
        scores[s + (num_instances * p)] = scores_on_training_[s] +
            weights[p] * score_instance_last_tree[s];
      }
    }

    #pragma omp parallel for
    for (unsigned int p = 0; p < weights.size(); ++p) {
      // Each thread computes the metric on some points of the window.
      // Thread p-th computes score on a part of the training_score vector
      // Operator & is used to obtain the first position of the sub-array
      metric_scores[p] = scorer->evaluate_dataset(
          dataset, &scores[num_instances * p]);
    }

    // Find the best metric score
    auto i_max_metric_score = std::max_element(metric_scores.cbegin(),
                                               metric_scores.cbegin() +
                                                   weights.size());
    auto idx = std::distance(metric_scores.cbegin(), i_max_metric_score);
    return weights[idx];

  } else if ( normalize_type == NormalizationType::CONTR ||
              normalize_type == NormalizationType::WCONTR) {

    double dropped_contribution = 0;
    for (auto t: dropped_trees) {
      dropped_contribution += scores_contribution_[t];
    }

    // weights contains the weights excluding the last added tree
    // so, size is the index of the new tree
    double contribution_last_tree = scores_contribution_[weights.size()];

    if (k <= 0)
      return shrinkage_;
    else
      return (dropped_contribution / contribution_last_tree) * shrinkage_;

  } else if (normalize_type == NormalizationType::LMART_ADAPTIVE) {

    return shrinkage_ / (rate_drop * ensemble_model_.get_size() + shrinkage_);

  }

  return 0;
}

int Dart::binary_search(std::vector<double>& array, double key) {

  int low = 0, high = (int) (array.size() - 1), midpoint = 0;

  while (low <= high) {

    midpoint = low + (high - low) / 2;
    if (key < array[midpoint] && (midpoint == 0 || key >= array[midpoint-1]) ) {
      return midpoint;
    } else if (key < array[midpoint])
      high = midpoint - 1;
    else
      low = midpoint + 1;
  }

  return -1;
}

void Dart::filter_out_zero_weighted_contributions(
    const std::vector<double>& weights) {

  size_t idx_curr = 0;
  for (size_t i = 0; i < weights.size(); ++i) {
    if (weights[i] > 0)
      scores_contribution_[idx_curr++] = scores_contribution_[i];
  }

  // erase previously saved values outside the new shape of contribution array
  for (size_t i = idx_curr; i < weights.size() && i < ntrees_; ++i) {
    scores_contribution_[i] = 0;
  }
}

int Dart::get_number_of_trees_to_dropout(
    std::vector<double>& performance_on_validation_per_iter,
    std::vector<double>& dropout_factor_per_iter,
    double best_on_validation,
    int dropped_before_cleaning) {

  double prob_skip_dropout = (double) rand() / (double) (RAND_MAX);
  int  model_size = (int) ensemble_model_.get_size() - dropped_before_cleaning;
  double trees_to_dropout = 0;
  if (prob_skip_dropout > skip_drop && model_size > 0) {

    if (adaptive_type == AdaptiveType::FIXED) {

      if (rate_drop >= 1) {
        // Avoid removing trees if the ensemble size is smaller than two times
        // the number of trees to remove
        if ((rate_drop * 2) <= model_size)
          trees_to_dropout = rate_drop;
      } else {
        trees_to_dropout = rate_drop * model_size;
      }

    } else if (adaptive_type == AdaptiveType::PLUS1_DIV2) {

      double last_dropout = dropout_factor_per_iter.back();
      if (performance_on_validation_per_iter.back() >= best_on_validation)
        trees_to_dropout = last_dropout / 2;
      else
        trees_to_dropout = last_dropout + 1;

    } else if (adaptive_type == AdaptiveType::PLUSHALF_DIV2) {

      double last_dropout = dropout_factor_per_iter.back();
      if (performance_on_validation_per_iter.back() >= best_on_validation)
        trees_to_dropout = last_dropout / 2;
      else
        trees_to_dropout = last_dropout + 0.5;

    } else if (adaptive_type == AdaptiveType::PLUSONETHIRD_DIV2) {

      double last_dropout = dropout_factor_per_iter.back();
      if (performance_on_validation_per_iter.back() >= best_on_validation)
        trees_to_dropout = last_dropout / 2;
      else
        trees_to_dropout = last_dropout + (1.0f / 3);

    } else if (adaptive_type == AdaptiveType::PLUSHALF_RESET) {

      double last_dropout = dropout_factor_per_iter.back();
      if (performance_on_validation_per_iter.back() >= best_on_validation)
        trees_to_dropout = 0;
      else
        trees_to_dropout = last_dropout + 0.5;

    } else if (adaptive_type == AdaptiveType::PLUSHALF_RESET_LB1_UB5) {

      double last_dropout = dropout_factor_per_iter.back();
      if (performance_on_validation_per_iter.back() >= best_on_validation)
        trees_to_dropout = 1;
      else
        trees_to_dropout = std::min(5.0, last_dropout + 0.5);

    } else if (adaptive_type == AdaptiveType::PLUSHALF_RESET_LB1_UB10) {

      double last_dropout = dropout_factor_per_iter.back();
      if (performance_on_validation_per_iter.back() >= best_on_validation)
        trees_to_dropout = 1;
      else
        trees_to_dropout = std::min(10.0, last_dropout + 0.5);

    } else if (adaptive_type == AdaptiveType::PLUSHALF_RESET_LB1_UBRD) {

      double last_dropout = dropout_factor_per_iter.back();
      if (performance_on_validation_per_iter.back() >= best_on_validation)
        trees_to_dropout = 1;
      else
        trees_to_dropout = std::min(rate_drop * model_size, last_dropout + 0.5);
    }
  }

  // At maximum we drop half of the trees
  trees_to_dropout = trees_to_dropout > model_size / 2
                     ? model_size / 2
                     : trees_to_dropout;
  dropout_factor_per_iter.push_back(trees_to_dropout);
  return (int) round(trees_to_dropout);
}

}  // namespace forests
}  // namespace learning
}  // namespace quickrank
