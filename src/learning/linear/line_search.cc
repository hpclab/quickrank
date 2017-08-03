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
#include "learning/linear/line_search.h"

#include <chrono>
#include <sstream>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <numeric>
#include "utils/strutils.h"

namespace quickrank {
namespace learning {
namespace linear {

const std::string LineSearch::NAME_ = "LINESEARCH";

LineSearch::LineSearch(unsigned int num_points,
                       double window_size,
                       double reduction_factor,
                       unsigned int max_iterations,
                       unsigned int max_failed_vali,
                       bool adaptive,
                       unsigned int last_only)
    : num_points_(num_points),
      window_size_(window_size),
      reduction_factor_(reduction_factor),
      max_iterations_(max_iterations),
      max_failed_vali_(max_failed_vali),
      adaptive_(adaptive),
      train_only_last_(last_only) {
}

LineSearch::LineSearch(const pugi::xml_document &model) {

  num_points_ = 0;
  window_size_ = 0.0;
  reduction_factor_ = 0.0;
  max_iterations_ = 0;
  max_failed_vali_ = 0;
  adaptive_ = true;

  //read (training) info
  //read (training) info
  pugi::xml_node model_info = model.child("ranker").child("info");
  pugi::xml_node model_ensemble = model.child("ranker").child("ensemble");

  num_points_ = model_info.child("num-samples").text().as_uint();
  window_size_ = model_info.child("window-size").text().as_double();
  reduction_factor_ = model_info.child("reduction-factor").text().as_double();
  max_iterations_ = model_info.child("max-iterations").text().as_uint();
  max_failed_vali_ = model_info.child("max-failed-vali").text().as_uint();

  if (model_info.child("adaptive"))
    adaptive_ = model_info.child("adaptive").text().as_bool();

  if (model_info.child("train-only-last"))
    train_only_last_ = model_info.child("train-only-last").text().as_uint();

  // Check if this is a full line search model or if it contains only the
  // preamble (for models which uses line search inside...)
  if (!model_ensemble.child("tree").child("index").empty()) {

    unsigned int max_feature = 0;
    for (const auto &tree: model_ensemble.children("tree")) {
      unsigned int feature = tree.child("index").text().as_uint();
      if (feature > max_feature) {
        max_feature = feature;
      }
    }

    std::vector<double>(max_feature, 0.0).swap(best_weights_);

    for (const auto &tree: model_ensemble.children("tree")) {
      unsigned int feature = tree.child("index").text().as_uint();
      double weight = tree.child("weight").text().as_double();
      best_weights_[feature - 1] = weight;
    }
  }
}

pugi::xml_document *LineSearch::get_xml_model() const {

  pugi::xml_document *doc = new pugi::xml_document();
  pugi::xml_node root = doc->append_child("ranker");

  pugi::xml_node info = root.append_child("info");

  info.append_child("type").text() = name().c_str();
  info.append_child("num-samples").text() = num_points_;
  info.append_child("window-size").text() = window_size_;
  info.append_child("reduction-factor").text() = reduction_factor_;
  info.append_child("max-iterations").text() = max_iterations_;
  info.append_child("max-failed-vali").text() = max_failed_vali_;
  info.append_child("adaptive").text() = adaptive_;
  info.append_child("train-only-last").text() = train_only_last_;

  std::stringstream ss;
  ss << std::setprecision(std::numeric_limits<double>::max_digits10);

  pugi::xml_node ensemble = root.append_child("ensemble");
  for (unsigned int i = 0; i < best_weights_.size(); i++) {

    ss << best_weights_[i];

    pugi::xml_node couple = ensemble.append_child("tree");
    couple.append_child("index").text() = i + 1;
    couple.append_child("weight").text() = ss.str().c_str();

    // reset ss
    ss.str(std::string());
  }

  return doc;
}

LineSearch::~LineSearch() {
}

std::ostream &LineSearch::put(std::ostream &os) const {
  os << "# Ranker: " << name() << std::endl
     << "# number of samples = " << num_points_ << std::endl
     << "# window size = " << window_size_ << std::endl
     << "# window reduction factor = " << reduction_factor_ << std::endl
     << "# number of max iterations = " << max_iterations_ << std::endl
     << "# number of fails on validation before exit = " << max_failed_vali_
     << std::endl
     << "# adaptive reduction factor = " << adaptive_ << std::endl;

  return os;
}

void LineSearch::learn(
    std::shared_ptr<quickrank::data::Dataset> training_dataset,
    std::shared_ptr<quickrank::data::Dataset> validation_dataset,
    std::shared_ptr<quickrank::metric::ir::Metric> scorer,
    size_t partial_save, const std::string output_basename) {

  auto begin = std::chrono::steady_clock::now();

  // We force num_points to be odd, so that the central point in step 1 is
  // included by default in searching the best weight for each feature
  unsigned int num_points = num_points_;
  if (num_points_ % 2)
    num_points--;

  std::cout << "# Training:" << std::endl;
  std::cout << std::fixed << std::setprecision(4);
  std::cout << "# -----------------------------------------------------";
  std::cout << std::endl;
  std::cout << "# iter. training validation   gain    window red_factor";
  std::cout << std::endl;
  std::cout << "# -----------------------------------------------------";
  std::cout << std::endl;

  const auto num_features = training_dataset->num_features();
  const auto num_train_instances = training_dataset->num_instances();

  std::vector<double> weights(num_features);
  std::vector<double> weights_prev(num_features);

  // If weights were not set before by calling the update_weights method,
  // set the starting weights to 1.0 by default
  if (best_weights_.empty())  {
    // Need the swap method because best_weights_ is unitialized
    std::vector<double>(num_features, 1.0f).swap(best_weights_);
  } else if (best_weights_.size() != num_features) {
    // The check on the number of features is needed because the line search model
    // could be reused on a different datasets (different size) w/o reset weights
    std::cerr << "Initial Line Search Weights does not correspond to datasets "
        "size" << std::endl;
    exit(EXIT_FAILURE);
  }

  print_weights(best_weights_, "LS Weights pre learning");

  // Copy the values of best_weights into weights and weights_prev (same size)
  std::copy(best_weights_.begin(), best_weights_.end(), weights.begin());
  std::copy(best_weights_.begin(), best_weights_.end(), weights_prev.begin());

  MetricScore best_metric_on_training = 0;
  MetricScore best_metric_on_validation = 0;

  // array of points in the window to be used to compute the metric
  std::vector<MetricScore> metric_scores(num_points + 1, 0.0);
  std::vector<Score> pre_sum(num_train_instances);
  std::vector<Score> training_score(num_train_instances * (num_points + 1));

  std::vector<Score> validation_score;
  if (validation_dataset)
    validation_score.resize(num_train_instances, 0.0);

  // compute training and validation scores using starting weights
  score(training_dataset->at(0, 0), num_train_instances, num_features,
        &weights[0], &training_score[0]);
  best_metric_on_training = scorer->evaluate_dataset(training_dataset,
                                                     &training_score[0]);
  std::cout << std::fixed << std::setprecision(4);
  std::cout << std::setw(7) << 0 << std::setw(9) << best_metric_on_training;
  if (validation_dataset) {
    score(validation_dataset->at(0, 0), validation_dataset->num_instances(),
          num_features, &weights[0], &validation_score[0]);
    best_metric_on_validation = scorer->evaluate_dataset(validation_dataset,
                                                         &validation_score[0]);
    std::cout << std::setw(9) << best_metric_on_validation << " *";
  }
  std::cout << std::endl;

  // window_size is the mean weight times the window_size_ factor
  double starting_window_size = std::accumulate(best_weights_.cbegin(),
                                                best_weights_.cend(),
                                                0.0) / best_weights_.size();
  // Multiply the average weight for the window size factor
  double window_size = starting_window_size * window_size_;

  unsigned int starting_feature_idx = 0;
  if (train_only_last_)
    starting_feature_idx = num_features - train_only_last_;

  // counter of sequential iterations without improvement on validation
  unsigned int count_failed_vali = 0;
  // loop for max_iterations_
  for (unsigned int i = 0; i < max_iterations_; i++) {

    // step1 length used to select points in the window
    double step1 = 2 * window_size / num_points;

    // Step 1: linear search on each feature (independently)
    for (unsigned int f = starting_feature_idx; f < num_features; f++) {

      // compute feature * weight for all the features different from f
      preCompute(training_dataset->at(0, 0), num_train_instances, num_features,
                 &pre_sum[0], &weights_prev[0], &training_score[0], f);

      // Compute the points (weights to try) related to feature f
      std::vector<double> points;
      points.reserve(num_points + 1); // don't know the exact number of points
      for (double point = weights_prev[f] - window_size;
           point <= weights_prev[f] + window_size; point += step1) {
        if (point >= 0)
          points.push_back(point);
      }

#pragma omp parallel for
      for (unsigned int s = 0; s < num_train_instances; ++s) {
        for (unsigned int p = 0; p < points.size(); ++p) {
          training_score[s + (num_train_instances * p)] =
              points[p] * training_dataset->at(s, f)[0] + pre_sum[s];
        }
      }

#pragma omp parallel for
      for (unsigned int p = 0; p < points.size(); ++p) {
        // Each thread computes the metric on some points of the window.
        // Thread p-th computes score on a part of the training_score vector
        // Operator & is used to obtain the first position of the sub-array
        metric_scores[p] = scorer->evaluate_dataset(
            training_dataset, &training_score[num_train_instances * p]);
      }

      // Find the best metric score
      auto i_max_metric_score = std::max_element(metric_scores.cbegin(),
                                                 metric_scores.cbegin() +
                                                     points.size());
      if (*i_max_metric_score > best_metric_on_training) {
        auto p = std::distance(metric_scores.cbegin(), i_max_metric_score);
        weights[f] = points[p];
      }
    }
    // end Step 1: linear search on each feature (independently)

    // Step 2: linear search on all features (between weights and weights_prev)

    // step2 length (different for each feature)
    std::vector<double> step2(num_features);
    std::transform(weights.begin(), weights.end(), weights_prev.begin(),
                   step2.begin(), [&](double curr, double prev) {
          return (curr - prev) / num_points;
        });

    // if step2 is a 0-vector, no way to improve in step2
    bool zeros = std::all_of(step2.begin(), step2.end(),
                             [](double s) { return s == 0; });
    double gain_on_training = 0;
    if (!zeros) {

#pragma omp parallel for
      for (unsigned int s = 0; s < num_train_instances; ++s) {
        for (unsigned int p = 0; p < num_points + 1; ++p) {
          Score score = 0;
          for (unsigned int f = 0; f < num_features; ++f) {
            score += (weights_prev[f] + step2[f] * p)
                * training_dataset->at(s, f)[0];
          }
          training_score[s + (num_train_instances * p)] = score;
        }
      }

#pragma omp parallel for
      for (unsigned int p = 0; p < num_points + 1; ++p) {
        // Each thread computes the metric on some points of the window.
        // Thread p-th computes score on a part of the training_score vector
        // Operator & is used to obtain the first position of the sub-array
        metric_scores[p] = scorer->evaluate_dataset(
            training_dataset, &training_score[num_train_instances * p]);
      }

      // Find the best metric score
      auto i_max_metric_score = std::max_element(metric_scores.cbegin(),
                                                 metric_scores.cend());
      if (*i_max_metric_score > best_metric_on_training) {
        auto p = std::distance(metric_scores.cbegin(), i_max_metric_score);

        // recompute the weights vector related to point p
        for (unsigned int f = 0; f < num_features; ++f) {
          weights[f] = (weights_prev[f] + step2[f] * p);
        }
        gain_on_training = *i_max_metric_score - best_metric_on_training;
        best_metric_on_training = *i_max_metric_score;
        // Set weights_prev to current weights for the next iteration
        weights_prev = weights;
      }

    } // end if zeros step2 vector

    std::cout << std::setw(7) << i + 1 << std::setw(9)
              << best_metric_on_training;

    auto cur_reduction_factor = reduction_factor_;
    if (adaptive_) {
      // TODO: fix, metric dependent (NDCG considered here)
      double max_gain = 0.005;
      // At most double the reduction factor (2 * reduction_factor_ > 1)
      double relative_gain =
          std::min((gain_on_training - max_gain) / max_gain, 1.);
      // At least half the reduction factor (0.5 * reduction_factor_)
      cur_reduction_factor = 1 + std::max(relative_gain, -0.5);
    }

    // check if there is validation_dataset
    if (validation_dataset) {
      // compute scores of validation documents
      for (unsigned int s = 0; s < validation_dataset->num_instances(); s++)
        validation_score[s] = std::inner_product(weights.cbegin(),
                                                 weights.cend(),
                                                 validation_dataset->at(s, 0),
                                                 (Score) 0.0);

      MetricScore metric_on_validation = scorer->evaluate_dataset(
          validation_dataset, &validation_score[0]);

      std::cout << std::setw(9) << metric_on_validation;
      if (metric_on_validation > best_metric_on_validation) {
        count_failed_vali = 0;  // reset to zero when validation improves
        best_metric_on_validation = metric_on_validation;
        best_weights_ = weights;
        std::cout << " *";
      } else {
        std::cout << "  ";
        if (++count_failed_vali >= max_failed_vali_) {
          std::cout << std::endl;
          break;
        }
      }
    } else {
      std::cout << std::setw(11) << "";
    }

    std::cout << " " << std::setw(7) << gain_on_training << " "
              << std::setw(8) << window_size << " "
              << std::setw(8) << cur_reduction_factor;

    std::cout << std::endl;
    window_size *= cur_reduction_factor;

    // if the cur window size is smaller than 1/10th of the original one, stop
    if (adaptive_ && window_size < starting_window_size / 10)
      break;

    if (partial_save != 0 and !output_basename.empty()
        and (i + 1) % partial_save == 0) {
      save(output_basename, i + 1);
    }
  }
  //end iterations

  // if validation dataset is missing, best_weights is found on training
  if (validation_dataset == NULL)
    best_weights_ = weights;

  print_weights(best_weights_, "LS Weights post learning");

  auto end = std::chrono::steady_clock::now();
  std::chrono::duration<double> elapsed = std::chrono::duration_cast<
      std::chrono::duration<double>>(end - begin);
  std::cout << std::endl;
  std::cout << "# \t Training time: " << std::setprecision(2) <<
            elapsed.count() << " seconds" << std::endl;
}

Score LineSearch::score_document(const Feature *d) const {
  Score score = 0;
  for (unsigned int k = 0; k < best_weights_.size(); k++) {
    score += best_weights_[k] * d[k];
  }
  return score;
}

bool LineSearch::update_weights(std::vector<double>& weights) {

  if (weights.size() != best_weights_.size()) {

    // copy the new weight vector, throwing away the old one (implicitly)
    best_weights_ = std::vector<double>(weights);

  } else {

    for (size_t k = 0; k < weights.size(); k++) {
      best_weights_[k] = weights[k];
    }
  }

  return true;
}

void LineSearch::preCompute(Feature *training_dataset, unsigned int num_samples,
                            unsigned int num_features, Score *pre_sum,
                            double *weights, Score *training_score,
                            unsigned int feature_exclude) {

#pragma omp parallel for
  for (unsigned int s = 0; s < num_samples; s++) {
    unsigned int offset_feature = s * num_features;
    pre_sum[s] = 0;
    training_score[s] = 0;
    // compute feature * weight for all the feature different from f
    for (unsigned int f = 0; f < num_features; f++) {
      training_score[s] += weights[f] * training_dataset[offset_feature + f];
    }
    pre_sum[s] = training_score[s] - (weights[feature_exclude] *
        training_dataset[offset_feature + feature_exclude]);
  }
}

void LineSearch::score(Feature *dataset, unsigned int num_samples,
                       unsigned int num_features, double *weights,
                       Score *scores) {

#pragma omp parallel for
  for (unsigned int s = 0; s < num_samples; s++) {
    unsigned int offset_feature = s * num_features;
    scores[s] = 0;
    // compute feature * weight for all the feature different from f
    for (unsigned int f = 0; f < num_features; f++) {
      scores[s] += weights[f] * dataset[offset_feature + f];
    }
  }
}

}  // namespace linear
}  // namespace learning
}  // namespace quickrank
