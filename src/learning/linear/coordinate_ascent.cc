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

#include <iostream>
#include <fstream>
#include <iomanip>
#include <cfloat>
#include <cmath>
#include <chrono>
#include <vector>
#include <numeric>
#include <algorithm>

#include <boost/property_tree/xml_parser.hpp>
#include <boost/foreach.hpp>

namespace quickrank {
namespace learning {
namespace linear {

void preCompute(Feature* training_dataset, unsigned int num_docs,
                unsigned int num_fx, Score* PreSum, double* weights,
                Score* MyTrainingScore, unsigned int i) {

#pragma omp parallel for
  for (unsigned int j = 0; j < num_docs; j++) {
    PreSum[j] = 0;
    MyTrainingScore[j] = 0;
    // compute feature*weight for all the feature different from i
    for (unsigned int k = 0; k < num_fx; k++) {
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

CoordinateAscent::CoordinateAscent(
    const boost::property_tree::ptree &info_ptree,
    const boost::property_tree::ptree &model_ptree) {

  num_samples_ = 0;
  window_size_ = 0.0;
  reduction_factor_ = 0.0;
  max_iterations_ = 0;
  max_failed_vali_ = 0;

  //read (training) info
  num_samples_ = info_ptree.get<unsigned int>("num-samples");
  window_size_ = info_ptree.get<double>("window-size");
  reduction_factor_ = info_ptree.get<double>("reduction-factor");
  max_iterations_ = info_ptree.get<unsigned int>("max-iterations");
  max_failed_vali_ = info_ptree.get<unsigned int>("max-failed-vali");

  unsigned int max_feature = 0;
  BOOST_FOREACH(const boost::property_tree::ptree::value_type &couple, model_ptree){

  if (couple.first =="couple") {
    unsigned int feature=couple.second.get<unsigned int>("feature");
    if(feature>max_feature) {
      max_feature=feature;
    }
  }
}

  std::vector<double>(max_feature, 0.0).swap(best_weights_);

  BOOST_FOREACH(const boost::property_tree::ptree::value_type &couple, model_ptree){
  if (couple.first =="couple") {
    int feature=couple.second.get<int>("feature");
    double weight=couple.second.get<double>("weight");
    best_weights_[feature-1]=weight;
  }
}
}

CoordinateAscent::~CoordinateAscent() {
}

std::ostream& CoordinateAscent::put(std::ostream& os) const {
  os << "# Ranker: " << name() << std::endl << "# number of samples = "
     << num_samples_ << std::endl << "# window size = " << window_size_
     << std::endl << "# window reduction factor = " << reduction_factor_
     << std::endl << "# number of max iterations = " << max_iterations_
     << std::endl << "# number of fails on validation before exit = "
     << max_failed_vali_ << std::endl;
  return os;
}

void CoordinateAscent::preprocess_dataset(
    std::shared_ptr<data::Dataset> dataset) const {
  if (dataset->format() != data::Dataset::HORIZ)
    dataset->transpose();
}

void CoordinateAscent::learn(
    std::shared_ptr<quickrank::data::Dataset> training_dataset,
    std::shared_ptr<quickrank::data::Dataset> validation_dataset,
    std::shared_ptr<quickrank::metric::ir::Metric> scorer,
    unsigned int partial_save, const std::string output_basename) {

  auto begin = std::chrono::steady_clock::now();
  double window_size = window_size_ / training_dataset->num_features();  //preserve original value of the window

      // Do some initialization
  preprocess_dataset(training_dataset);
  if (validation_dataset)
    preprocess_dataset(validation_dataset);

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
  unsigned int count_failed_vali = 0;
  // loop for max_iterations_
  for (unsigned int b = 0; b < max_iterations_; b++) {
    MetricScore metric_on_training = 0;

    double step = 2 * window_size / num_samples_;  // step to select points in the window
    for (unsigned int i = 0; i < num_features; i++) {
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
      for (unsigned int p = 0; p < points.size(); p++) {
        //loop to add partial scores to the total score of the feature i
        for (unsigned int j = 0; j < n_train_instances; j++) {
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
                      [normalized_sum](double &x) {x/=normalized_sum;});
      }

    }  // end for i

    std::cout << std::setw(7) << b + 1 << std::setw(9) << metric_on_training;

    // check if there is validation_dataset
    if (validation_dataset) {
      //compute scores of validation documents
      for (unsigned int j = 0; j < validation_dataset->num_instances(); j++)
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

void CoordinateAscent::score_dataset(std::shared_ptr<data::Dataset> dataset,
                                     Score* scores) const {
  preprocess_dataset(dataset);

  for (unsigned int q = 0; q < dataset->num_queries(); q++) {
    std::shared_ptr<data::QueryResults> r = dataset->getQueryResults(q);
    score_query_results(r, scores, dataset->num_features());
    scores += r->num_results();
  }
}

void CoordinateAscent::score_query_results(
    std::shared_ptr<data::QueryResults> results, Score* scores,
    unsigned int offset) const {
  const quickrank::Feature* d = results->features();
  for (unsigned int i = 0; i < results->num_results(); i++) {
    scores[i] = score_document(d, offset);
    d += offset;
  }
}

// assumes vertical dataset
Score CoordinateAscent::score_document(const quickrank::Feature* d,
                                       const unsigned int offset) const {
  Score score = 0;
  for (unsigned int k = 0; k < offset; k++) {
    score += best_weights_[k] * d[k];
  }
  return score;

}

std::ofstream& CoordinateAscent::save_model_to_file(std::ofstream& os) const {
  // write ranker description
  os << "\t<info>" << std::endl;
  os << "\t\t<type>" << name() << "</type>" << std::endl;
  os << "\t\t<num-samples>" << num_samples_ << "</num-samples>" << std::endl;
  os << "\t\t<window-size>" << window_size_ << "</window-size>" << std::endl;
  os << "\t\t<reduction-factor>" << reduction_factor_ << "</reduction-factor>"
     << std::endl;
  os << "\t\t<max-iterations>" << max_iterations_ << "</max-iterations>"
     << std::endl;
  os << "\t\t<max-failed-vali>" << max_failed_vali_ << "</max-failed-vali>"
     << std::endl;
  os << "\t</info>" << std::endl;

  os << "\t<ensemble>" << std::endl;
  auto old_precision = os.precision();
  os.setf(std::ios::floatfield, std::ios::fixed);
  for (unsigned int i = 0; i < best_weights_.size(); i++) {
    os << "\t\t<couple>" << std::endl;
    os << std::setprecision(3);
    os << "\t\t\t<feature>" << i + 1 << "</feature>" << std::endl;
    os << std::setprecision(std::numeric_limits<quickrank::Score>::digits10);
    os << "\t\t\t<weight>" << best_weights_[i] << "</weight>" << std::endl;
    os << "\t\t</couple>" << std::endl;
  }
  os << "\t</ensemble>" << std::endl;
  os << std::setprecision(old_precision);
  return os;
}

}  // namespace linear
}  // namespace learning
}  // namespace quickrank
