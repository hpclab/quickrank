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
#include "learning/custom/custom_ltr.h"

#include <iostream>
#include <fstream>
#include <iomanip>
#include <cfloat>
#include <cmath>

namespace quickrank {
namespace learning {

const std::string CustomLTR::NAME_ = "CUSTOM";

CustomLTR::CustomLTR() {
}

CustomLTR::~CustomLTR() {
}

std::ostream& CustomLTR::put(std::ostream& os) const {
  os << "# Ranker: " << name() << std::endl;
  return os;
}

void CustomLTR::preprocess_dataset(
    std::shared_ptr<data::Dataset> dataset) const {
  if (dataset->format() != data::Dataset::HORIZ)
    dataset->transpose();
}

void CustomLTR::learn(
    std::shared_ptr<quickrank::data::Dataset> training_dataset,
    std::shared_ptr<quickrank::data::Dataset> validation_dataset,
    std::shared_ptr<quickrank::metric::ir::Metric> scorer,
    unsigned int partial_save, const std::string output_basename) {

  // Do some initialization
  preprocess_dataset(training_dataset);
  preprocess_dataset(validation_dataset);

  std::cout << "# Training..." << std::endl;
  std::cout << std::fixed << std::setprecision(4);

  // allocate scores
  Score* training_scores = new Score[training_dataset->num_instances()];
  Score* validation_scores = new Score[validation_dataset->num_instances()];

  // set scores equal to fixed value
  for (unsigned int i = 0; i < training_dataset->num_instances(); i++)
    training_scores[i] = FIXED_SCORE;

  MetricScore metric_on_training = scorer->evaluate_dataset(training_dataset,
                                                            training_scores);

  std::cout << *scorer << " on training: " << metric_on_training << std::endl;

  for (unsigned int i = 0; i < validation_dataset->num_instances(); i++)
    validation_scores[i] = FIXED_SCORE;

  MetricScore metric_on_validation = scorer->evaluate_dataset(
      validation_dataset, validation_scores);

  std::cout << *scorer << " on validation: " << metric_on_validation
            << std::endl;

  std::cout << "# Training completed." << std::endl;

  delete[] training_scores;
  delete[] validation_scores;
}

// assumes vertical dataset
Score CustomLTR::score_document(const quickrank::Feature* d,
                                const unsigned int offset) const {
  return FIXED_SCORE;
}

std::ofstream& CustomLTR::save_model_to_file(std::ofstream& os) const {
  // write ranker description
  os << *this;
  // save xml model
  // TODO: Save model to file
  return os;
}

}  // namespace learning
}  // namespace quickrank
