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
 *  - Andrea Battistini (andreabattistini@hotmail.com)
 *  - Chiara Pierucci (chiarapierucci14@gmail.com)
 *  - Claudio Lucchese (claudio.lucchese@isti.cnr.it)
 */
#include "learning/linear/coordinate_ascent.h"

#include <fstream>
#include <iomanip>
#include <cmath>
#include <chrono>
#include <numeric>
#include <sstream>

namespace quickrank {
namespace learning {
namespace linear {

void preCompute(Feature *training_dataset, size_t num_docs,
                size_t num_fx, Score *PreSum, double *weights,
                Score *MyTrainingScore, size_t i) {

#pragma omp parallel for
  for (size_t j = 0; j < num_docs; j++) {
    PreSum[j] = 0;
    MyTrainingScore[j] = 0;
    // compute feature*weight for all the feature different from i
    for (size_t k = 0; k < num_fx; k++) {
      MyTrainingScore[j] += weights[k] * training_dataset[j * num_fx + k];
    }
    PreSum[j] = MyTrainingScore[j]
        - (weights[i] * training_dataset[j * num_fx + i]);
  }
}

const std::string CoordinateAscent::NAME_ = "COORDASC";

CoordinateAscent::CoordinateAscent(unsigned int num_points, double window_size,
                                   double reduction_factor,
                                   unsigned int max_iterations,
                                   unsigned int max_failed_vali)
    : num_samples_(num_points),
      window_size_(window_size),
      reduction_factor_(reduction_factor),
      max_iterations_(max_iterations),
      max_failed_vali_(max_failed_vali) {
}

CoordinateAscent::CoordinateAscent(const pugi::xml_document &model) {

  num_samples_ = 0;
  window_size_ = 0.0;
  reduction_factor_ = 0.0;
  max_iterations_ = 0;
  max_failed_vali_ = 0;

  //read (training) info
  pugi::xml_node model_info = model.child("ranker").child("info");
  pugi::xml_node model_ensemble = model.child("ranker").child("model");

  num_samples_ = model_info.child("num-samples").text().as_uint();
  window_size_ = model_info.child("window-size").text().as_double();
  reduction_factor_ = model_info.child("reduction-factor").text().as_double();
  max_iterations_ = model_info.child("max-iterations").text().as_uint();
  max_failed_vali_ = model_info.child("max-failed-vali").text().as_uint();

  unsigned int max_feature = 0;
  for (const auto &feature: model_ensemble.children("feature")) {
    unsigned int featureId = feature.attribute("id").as_uint();
    if (featureId > max_feature) {
      max_feature = featureId;
    }
  }

  std::vector<double>(max_feature, 0.0).swap(best_weights_);

  for (const auto &feature: model_ensemble.children("feature")) {
    unsigned int featureId = feature.attribute("id").as_uint();
    double weight = feature.attribute("weight").as_double();
    best_weights_[featureId - 1] = weight;
  }
}

CoordinateAscent::~CoordinateAscent() {
}

std::ostream &CoordinateAscent::put(std::ostream &os) const {
  os << "# Ranker: " << name() << std::endl << "# number of samples = "
     << num_samples_ << std::endl << "# window size = " << window_size_
     << std::endl << "# window reduction factor = " << reduction_factor_
     << std::endl << "# number of max iterations = " << max_iterations_
     << std::endl << "# number of fails on validation before exit = "
     << max_failed_vali_ << std::endl;
  return os;
}


void CoordinateAscent::learn(
    std::shared_ptr<quickrank::data::Dataset> training_dataset,
    std::shared_ptr<quickrank::data::Dataset> validation_dataset,
    std::shared_ptr<quickrank::metric::ir::Metric> scorer,
    size_t partial_save, const std::string output_basename) {

  auto begin = std::chrono::steady_clock::now();
  double window_size = window_size_
      / training_dataset->num_features();  //preserve original value of the window

  std::cout << "# Training:" << std::endl;
  std::cout << std::fixed << std::setprecision(4);
  std::cout << "# --------------------------" << std::endl;
  std::cout << "# iter. training validation" << std::endl;
  std::cout << "# --------------------------" << std::endl;

  // initialize weights and best_weights a 1/n
  const auto num_features = training_dataset->num_features();
  const auto n_train_instances = training_dataset->num_instances();

  std::vector<double> weights(num_features, 1.0 / num_features);
  std::vector<double>(num_features, 1.0 / num_features).swap(best_weights_);

  // array of points in the window to be used to compute NDCG 
  std::vector<MetricScore> MyNDCGs(num_samples_ + 1);
  MetricScore Bestmetric_on_validation = 0;
  std::vector<Score> PreSum(n_train_instances);
  std::vector<Score> MyTrainingScore(n_train_instances * (num_samples_ + 1));

  std::vector<Score> MyValidationScore;
  if (validation_dataset)
    MyValidationScore.resize(n_train_instances);

  // counter of sequential iterations without improvement on validation
  size_t count_failed_vali = 0;
  // loop for max_iterations_
  for (size_t b = 0; b < max_iterations_; b++) {
    MetricScore metric_on_training = 0;

    double step =
        2 * window_size / num_samples_;  // step to select points in the window
    for (size_t i = 0; i < num_features; i++) {
      // compute feature*weight for all the feature different from i
      preCompute(training_dataset->at(0, 0), n_train_instances, num_features,
                 &PreSum[0], &weights[0], &MyTrainingScore[0], i);

      metric_on_training = scorer->evaluate_dataset(training_dataset,
                                                    &MyTrainingScore[0]);

      std::vector<double> points;
      points.reserve(num_samples_ + 1);
      for (double lower_bound = weights[i] - window_size;
           lower_bound <= weights[i] + window_size; lower_bound += step) {
        if (lower_bound >= 0)
          points.push_back(lower_bound);
      }

#pragma omp parallel for
      for (size_t p = 0; p < points.size(); p++) {
        //loop to add partial scores to the total score of the feature i
        for (size_t j = 0; j < n_train_instances; j++) {
          MyTrainingScore[j + (n_train_instances * p)] = points[p]
              * training_dataset->at(j, i)[0] + PreSum[j];
        }
        // each thread computes NDCG on some points of the window
        // scorer gets a part of array MyTrainingScore for the thread p-th
        // Operator & is used to obtain the first position of the sub-array
        MyNDCGs[p] = scorer->evaluate_dataset(
            training_dataset, &MyTrainingScore[n_train_instances * p]);
      }
      // End parallel

      // Find the best NDCG
      auto i_max_ndcg = std::max_element(MyNDCGs.cbegin(), MyNDCGs.cend());
      if (*i_max_ndcg > metric_on_training) {
        auto p = std::distance(MyNDCGs.cbegin(), i_max_ndcg);
        weights[i] = points[p];
        metric_on_training = *i_max_ndcg;
        // normalize
        double normalized_sum = std::accumulate(weights.cbegin(),
                                                weights.cend(), 0.0);
        std::for_each(weights.begin(), weights.end(),
                      [normalized_sum](double &x) { x /= normalized_sum; });
      }

    }  // end for i

    std::cout << std::setw(7) << b + 1 << std::setw(9) << metric_on_training;

    // check if there is validation_dataset
    if (validation_dataset) {
      //compute scores of validation documents
      for (size_t j = 0; j < validation_dataset->num_instances(); j++)
        MyValidationScore[j] = std::inner_product(weights.cbegin(),
                                                  weights.cend(),
                                                  validation_dataset->at(j, 0),
                                                  (Score) 0.0);

      MetricScore metric_on_validation = scorer->evaluate_dataset(
          validation_dataset, &MyValidationScore[0]);

      std::cout << std::setw(9) << metric_on_validation;
      if (metric_on_validation > Bestmetric_on_validation) {
        count_failed_vali = 0;  //reset to zero when validation improves
        Bestmetric_on_validation = metric_on_validation;
        best_weights_ = weights;
        std::cout << " *";
      } else {
        count_failed_vali++;
        if (count_failed_vali >= max_failed_vali_) {
          std::cout << std::endl;
          break;
        }
      }
    }

    std::cout << std::endl;
    window_size *= reduction_factor_;
  }
  //end iterations

  //if there is no validation dataset get the weights of training as best_weights 
  if (validation_dataset == NULL)
    best_weights_ = weights;

  auto end = std::chrono::steady_clock::now();
  std::chrono::duration<double> elapsed = std::chrono::duration_cast<
      std::chrono::duration<double>>(end - begin);
  std::cout << std::endl;
  std::cout << "# \t Training time: " << std::setprecision(2) << elapsed.count()
            << " seconds" << std::endl;

}

Score CoordinateAscent::score_document(const Feature *d) const {
  Score score = 0;
  for (size_t k = 0; k < best_weights_.size(); k++) {
    score += best_weights_[k] * d[k];
  }
  return score;
}

bool CoordinateAscent::update_weights(std::vector<double>& weights) {

  if (weights.size() != best_weights_.size())
    return false;

  for (size_t k = 0; k < weights.size(); k++) {
    best_weights_[k] = weights[k];
  }

  return true;
}

pugi::xml_document *CoordinateAscent::get_xml_model() const {

  pugi::xml_document *doc = new pugi::xml_document();
  pugi::xml_node root = doc->append_child("ranker");

  pugi::xml_node info = root.append_child("info");

  info.append_child("type").text() = name().c_str();
  info.append_child("num-samples").text() = num_samples_;
  info.append_child("window-size").text() = window_size_;
  info.append_child("reduction-factor").text() = reduction_factor_;
  info.append_child("max-iterations").text() = max_iterations_;
  info.append_child("max-failed-vali").text() = max_failed_vali_;

  std::stringstream ss;
  ss << std::setprecision(std::numeric_limits<double>::max_digits10);

  pugi::xml_node model = root.append_child("model");
  for (size_t i = 0; i < best_weights_.size(); i++) {

    ss << best_weights_[i];

    pugi::xml_node feature = model.append_child("feature");

    feature.append_attribute("id") = i + 1;
    feature.append_attribute("weight") = ss.str().c_str();

    // reset ss
    ss.str(std::string());
  }

  return doc;
}

}  // namespace linear
}  // namespace learning
}  // namespace quickrank
