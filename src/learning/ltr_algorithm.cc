#include <fstream>

#include "learning/ltr_algorithm.h"
#include "utils/mergesorter.h"

#ifdef _OPENMP
#include <omp.h>
#else
#include "utils/omp-stubs.h"
#endif

namespace quickrank {
namespace learning {


void LTR_Algorithm::score_dataset(std::shared_ptr<data::Dataset> dataset,
                                  Score* scores) const {
  preprocess_dataset(dataset);
  for (unsigned int q = 0; q < dataset->num_queries(); q++) {
    std::shared_ptr<data::QueryResults> r = dataset->getQueryResults(q);
    score_query_results(r, scores, dataset->num_instances());
    scores += r->num_results();
  }
}

// assumes vertical dataset
// offset to next feature of the same instance
void LTR_Algorithm::score_query_results(
    std::shared_ptr<data::QueryResults> results,
    Score* scores, unsigned int offset) const {
  const quickrank::Feature* d = results->features();
  for (unsigned int i = 0; i < results->num_results(); i++) {
    scores[i] = score_document(d, offset);
    d++;
  }
}

// assumes vertical dataset
Score LTR_Algorithm::score_document(const quickrank::Feature* d,
                                    const unsigned int offset) const {
  return 0.0;
}

void LTR_Algorithm::save(std::string output_basename, int iteration) const {
  if (!output_basename.empty()) {
    std::string filename = output_basename;
    if (iteration != -1)
      filename += iteration + ".xml";
    std::ofstream output_file;
    output_file.open(filename);
    save_model_to_file(output_file);
    output_file.close();
  }
}

LTR_Algorithm* LTR_Algorithm::load_model_from_file(std::string model_filename) {
  if (model_filename.empty())
    return NULL;

  std::cout<<"ci starebbe bene un bestemmione!"<< std::endl;
  return NULL;

}

}  // namespace learning
}  // namespace quickrank
