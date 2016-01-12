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
#include "pruning/ensemble_pruning.h"

#include <fstream>
#include <iomanip>
#include <chrono>

#include <boost/property_tree/xml_parser.hpp>
#include <io/svml.h>
#include <plotcompat.h>

namespace quickrank {
namespace pruning {

const std::string EnsemblePruning::NAME_ = "EPRUNING";

const std::vector<std::string> EnsemblePruning::pruningMethodName = {
  "RANDOM", "LOW_WEIGHTS", "SKIP", "LAST", "QUALITY_LOSS", "AGGR_SCORE"
};

EnsemblePruning::EnsemblePruning(PruningMethod pruning_method,
                                 double pruning_rate)
    : pruning_rate_(pruning_rate),
      pruning_method_(pruning_method),
      lineSearch_() {
}

EnsemblePruning::EnsemblePruning(std::string pruning_method,
                                 double pruning_rate) :
    pruning_rate_(pruning_rate),
    pruning_method_(getPruningMethod(pruning_method)),
    lineSearch_() {
}

EnsemblePruning::EnsemblePruning(std::string pruning_method,
                                 double pruning_rate,
                                 std::shared_ptr<learning::linear::LineSearch> lineSearch) :
    pruning_rate_(pruning_rate),
    pruning_method_(getPruningMethod(pruning_method)),
    lineSearch_(lineSearch)  {
}

EnsemblePruning::EnsemblePruning(const boost::property_tree::ptree &info_ptree,
                                 const boost::property_tree::ptree &model_ptree)
{
  pruning_rate_ = info_ptree.get <double> ("pruning-rate");
  auto pruning_method_name = info_ptree.get <std::string> ("pruning-method");
  pruning_method_ = getPruningMethod(pruning_method_name);

  unsigned int max_feature = 0;
  for (const boost::property_tree::ptree::value_type& tree: model_ptree) {
    if (tree.first == "tree") {
      unsigned int feature = tree.second.get<unsigned int>("index");
      if (feature > max_feature) {
        max_feature = feature;
      }
    }
  }

  estimators_to_select_ = 0;
  std::vector<double>(max_feature, 0.0).swap(weights_);
  for (const boost::property_tree::ptree::value_type& tree: model_ptree) {
    if (tree.first == "tree") {
      int feature = tree.second.get<int>("index");
      double weight = tree.second.get<double>("weight");
      weights_[feature - 1] = weight;
      if (weight > 0)
        estimators_to_select_++;
    }
  }
}

EnsemblePruning::~EnsemblePruning() {
}

std::ostream& EnsemblePruning::put(std::ostream &os) const {
  os << "# Ranker: " << name() << std::endl
    << "# pruning rate = " << pruning_rate_ << std::endl
    << "# pruning method = " << getPruningMethod(pruning_method_) << std::endl;
  if (lineSearch_)
    os << "# Line Search Parameters: " << std::endl << *lineSearch_ <<
        std::endl;
  else
    os << "# No Line Search" << std::endl;
  return os << std::endl;
}

void EnsemblePruning::preprocess_dataset(
    std::shared_ptr<data::Dataset> dataset) const {

  if (dataset->format() != data::Dataset::HORIZ)
    dataset->transpose();
}

void EnsemblePruning::learn(
    std::shared_ptr<quickrank::data::Dataset> training_dataset,
    std::shared_ptr<quickrank::data::Dataset> validation_dataset,
    std::shared_ptr<quickrank::metric::ir::Metric> scorer,
    unsigned int partial_save, const std::string output_basename) {

  auto begin = std::chrono::steady_clock::now();

  // Do some initialization
  preprocess_dataset(training_dataset);
  if (validation_dataset)
    preprocess_dataset(validation_dataset);

  if (pruning_rate_ < 1)
    estimators_to_select_ = pruning_rate_ * training_dataset->num_features();
  else {
    if (estimators_to_select_ > training_dataset->num_features())
      return; // nothing to do
    estimators_to_select_ = pruning_rate_;
  }
  std::vector<double>(training_dataset->num_features(), 1.0).swap(weights_);

  // compute training and validation scores using starting weights

  std::vector<Score> training_score(training_dataset->num_instances());
  score(training_dataset.get(), &training_score[0]);

  auto init_metric_on_training = scorer->evaluate_dataset(training_dataset,
                                                     &training_score[0]);

  std::cout << std::endl;
  std::cout << "# Initial metric without pruning:" << std::endl;
  std::cout << std::fixed << std::setprecision(4);
  std::cout << "# --------------------------" << std::endl;
  std::cout << "#       training validation" << std::endl;
  std::cout << "# --------------------------" << std::endl;
  std::cout << std::setw(7) << " " << std::setw(9) << init_metric_on_training;
  if (validation_dataset) {
    std::vector<Score> validation_score(validation_dataset->num_instances());
    score(validation_dataset.get(), &validation_score[0]);
    auto init_metric_on_validation = scorer->evaluate_dataset(
        validation_dataset, &validation_score[0]);
    std::cout << std::setw(9) << init_metric_on_validation
      << std::endl << std::endl;
  }

  // TODO: Implement remainings ensemble pruning strategies
  switch (pruning_method_) {
    case PruningMethod::RANDOM: {
      random_pruning(training_dataset);
      break;
    }
    default:
      throw std::invalid_argument("pruning method still not implemented");
      break;
  }

  if (lineSearch_) {
    // Filter the dataset by deleting the weight-0 features
    std::shared_ptr<data::Dataset> filtered_training_dataset;
    std::shared_ptr<data::Dataset> filtered_validation_dataset;

    filtered_training_dataset = filter_dataset(training_dataset);
    if (validation_dataset)
      filtered_validation_dataset = filter_dataset(validation_dataset);

    // Run the line search algorithm
    lineSearch_->learn(filtered_training_dataset, filtered_validation_dataset,
                       scorer, partial_save, output_basename);
  }

  auto end = std::chrono::steady_clock::now();
  std::chrono::duration<double> elapsed = std::chrono::duration_cast<
      std::chrono::duration<double>>(end - begin);
  std::cout << std::endl;
  std::cout << "# \t Training time: " << std::setprecision(2) <<
      elapsed.count() << " seconds" << std::endl;

}

Score EnsemblePruning::score_document(const Feature *d,
                                      const unsigned int next_fx_offset) const {
  // next_fx_offset is ignored as it is equal to 1 for horizontal dataset
  Score score = 0;
  for (unsigned int k = 0; k < weights_.size(); k++) {
    score += weights_[k] * d[k];
  }
  return score;
}

std::ofstream& EnsemblePruning::save_model_to_file(std::ofstream &os) const {
  // write ranker description
  os << "\t<info>" << std::endl;
  os << "\t\t<type>" << name() << "</type>" << std::endl;
  os << "\t\t<pruning-method>" << getPruningMethod(pruning_method_) <<
      "</pruning-method>" << std::endl;
  os << "\t\t<pruning-rate>" << pruning_rate_ << "</pruning-rate>" << std::endl;
  os << "\t</info>" << std::endl;

  os << "\t<ensemble>" << std::endl;
  auto old_precision = os.precision();
  os.setf(std::ios::floatfield, std::ios::fixed);
  for (unsigned int i = 0; i < weights_.size(); i++) {
    os << "\t\t<tree>" << std::endl;
    os << std::setprecision(3);
    os << "\t\t\t<index>" << i + 1 << "</index>" << std::endl;
    os << std::setprecision(std::numeric_limits<quickrank::Score>::digits10);
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
    // compute feature * weight for all the feature different from f
    for (unsigned int f = 0; f < dataset->num_features(); f++) {
      scores[s] += features[offset_feature + f];
    }
  }
}

void EnsemblePruning::random_pruning(std::shared_ptr<data::Dataset> dataset) {

  unsigned int num_features = dataset->num_features();

  /* initialize random seed: */
  srand (time(NULL));

  unsigned int selected = 0;
  while (selected < estimators_to_select_) {
    unsigned int index = rand() % num_features;
    if (weights_[index] > 0) {
      selected++;
      weights_[index] = 0;
    }
  }
}

std::shared_ptr<data::Dataset> EnsemblePruning::filter_dataset(
      std::shared_ptr<data::Dataset> dataset) const {


  data::Dataset* filt_dataset = new data::Dataset(dataset->num_instances(),
                                                  estimators_to_select_);

  // allocate feature vector
  std::vector<Feature> featureCleaned(estimators_to_select_);
  unsigned int skipped;

  if (dataset->format() == dataset->VERT)
    dataset->transpose();

  for (unsigned int q = 0; q < dataset->num_queries(); q++) {
    std::shared_ptr<data::QueryResults> results = dataset->getQueryResults(q);
    const Feature* features = results->features();
    const Label* labels = results->labels();

    for (unsigned int r = 0; r < results->num_results(); r++) {
      skipped = 0;
      for (unsigned int f = 0; f < dataset->num_features(); f++) {
        if (weights_[f] == 0) {
          skipped++;
          continue;
        } else {
          featureCleaned[f - skipped] = features[f];
        }
      }
      features += dataset->num_features();
      filt_dataset->addInstance(q, labels[r], featureCleaned);
    }
  }

  return std::shared_ptr<data::Dataset>(filt_dataset);
}

}  // namespace pruning
}  // namespace quickrank
