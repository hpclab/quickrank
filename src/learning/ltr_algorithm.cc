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


void LTR_Algorithm::score_dataset(std::shared_ptr<quickrank::data::Dataset> dataset,
                                  quickrank::Score* scores) const {
  if (dataset->format() != quickrank::data::Dataset::VERT)
    dataset->transpose();
  for (unsigned int q = 0; q < dataset->num_queries(); q++) {
    std::shared_ptr<quickrank::data::QueryResults> r = dataset->getQueryResults(q);
    score_query_results(r, scores, dataset->num_instances());
    scores += r->num_results();
  }
}

// assumes vertical dataset
// offset to next feature of the same instance
void LTR_Algorithm::score_query_results(
    std::shared_ptr<quickrank::data::QueryResults> results,
    quickrank::Score* scores, unsigned int offset) const {
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

}  // namespace learning
}  // namespace quickrank
