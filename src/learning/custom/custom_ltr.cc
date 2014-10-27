#include "learning/custom/custom_ltr.h"

#include <iostream>
#include <fstream>
#include <iomanip>
#include <cfloat>
#include <cmath>

#include "utils/radix.h"
#include "utils/qsort.h"
#include "utils/mergesorter.h"

namespace quickrank {
namespace learning {

CustomLTR::CustomLTR() {
}

CustomLTR::~CustomLTR() {
}

std::ostream& CustomLTR::put(std::ostream& os) const {
  os << "# Ranker: Custom" << std::endl;
  return os;
}

void CustomLTR::preprocess_dataset(std::shared_ptr<data::Dataset> dataset) const {
  if (dataset->format() != data::Dataset::HORIZ)
    dataset->transpose();
}


void CustomLTR::learn(std::shared_ptr<quickrank::data::Dataset> training_dataset,
                       std::shared_ptr<quickrank::data::Dataset> validation_dataset,
                       std::shared_ptr<quickrank::metric::ir::Metric> scorer, unsigned int partial_save,
                       const std::string output_basename) {

  // Do some initialization
  preprocess_dataset(training_dataset);
  preprocess_dataset(validation_dataset);

  std::cout << "# Training..." << std::endl;
  std::cout << std::fixed << std::setprecision(4);

  // allocate scores
  Score* training_scores = new Score[training_dataset->num_instances()];
  Score* validation_scores = new Score[validation_dataset->num_instances()];

  // set scores equal to fixed value
  for (unsigned int i=0; i<training_dataset->num_instances(); i++)
    training_scores[i] = FIXED_SCORE;

  MetricScore metric_on_training = scorer->evaluate_dataset(
      training_dataset, training_scores);

  std::cout << *scorer << " on training: " << metric_on_training << std::endl;


  for (unsigned int i=0; i<validation_dataset->num_instances(); i++)
    validation_scores[i] = FIXED_SCORE;

  MetricScore metric_on_validation = scorer->evaluate_dataset(
      validation_dataset, validation_scores);

  std::cout << *scorer << " on validation: " << metric_on_validation << std::endl;

  std::cout << "# Training completed." << std::endl;

  delete [] training_scores;
  delete [] validation_scores;
}

void CustomLTR::score_dataset(std::shared_ptr<data::Dataset> dataset,
                                  Score* scores) const {
  preprocess_dataset(dataset);
  for (unsigned int q = 0; q < dataset->num_queries(); q++) {
    std::shared_ptr<data::QueryResults> r = dataset->getQueryResults(q);
    score_query_results(r, scores, 1);
    scores += r->num_results();
  }
}

// assumes vertical dataset
// offset to next feature of the same instance
void CustomLTR::score_query_results(
    std::shared_ptr<data::QueryResults> results,
    Score* scores, unsigned int offset) const {
  const quickrank::Feature* d = results->features();
  for (unsigned int i = 0; i < results->num_results(); i++) {
    scores[i] = score_document(d, 1);
    d++;
  }
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
