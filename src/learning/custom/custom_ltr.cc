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

#include <fstream>
#include <iomanip>
#include <cmath>

namespace quickrank {
namespace learning {

const std::string CustomLTR::NAME_ = "CUSTOM";

CustomLTR::CustomLTR() {
}

CustomLTR::~CustomLTR() {
}

std::ostream &CustomLTR::put(std::ostream &os) const {
  os << "# Ranker: " << name() << std::endl;
  return os;
}

void CustomLTR::learn(
    std::shared_ptr<quickrank::data::Dataset> training_dataset,
    std::shared_ptr<quickrank::data::Dataset> validation_dataset,
    std::shared_ptr<quickrank::metric::ir::Metric> scorer,
    size_t partial_save, const std::string output_basename) {

  std::cout << "# Training..." << std::endl;
  std::cout << std::fixed << std::setprecision(4);

  // allocate scores
  Score *training_scores = new Score[training_dataset->num_instances()];
  Score *validation_scores = new Score[validation_dataset->num_instances()];

  // set scores equal to fixed value
  for (size_t i = 0; i < training_dataset->num_instances(); i++)
    training_scores[i] = FIXED_SCORE;

  MetricScore metric_on_training = scorer->evaluate_dataset(training_dataset,
                                                            training_scores);

  std::cout << *scorer << " on training: " << metric_on_training << std::endl;

  for (size_t i = 0; i < validation_dataset->num_instances(); i++)
    validation_scores[i] = FIXED_SCORE;

  MetricScore metric_on_validation = scorer->evaluate_dataset(
      validation_dataset, validation_scores);

  std::cout << *scorer << " on validation: " << metric_on_validation
            << std::endl;

  std::cout << "# Training completed." << std::endl;

  delete[] training_scores;
  delete[] validation_scores;
}

Score CustomLTR::score_document(const quickrank::Feature *d) const {
  return FIXED_SCORE;
}

pugi::xml_document *CustomLTR::get_xml_model() const {
  pugi::xml_document *doc = new pugi::xml_document();
  doc->set_name("ranker");

  pugi::xml_node info = doc->append_child("info");

  info.append_child("type").text() = name().c_str();

  return doc;
}

}  // namespace learning
}  // namespace quickrank
