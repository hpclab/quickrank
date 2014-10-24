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

// NOTE this replaces a "lot" of methods used in lmart, ranker, evaluator
float LTR_Algorithm::compute_score(LTR_VerticalDataset *samples,
                                   quickrank::metric::ir::Metric* scorer) {
  const unsigned int nrankedlists = samples->get_nrankedlists();
  //unsigned int * const rloffsets = samples->get_rloffsets();
  float * const * const featurematrix = samples->get_fmatrix();
  float score = 0.0f;
#pragma omp parallel for reduction(+:score)
  for (unsigned int i = 0; i < nrankedlists; ++i) {
    ResultList ql = samples->get_qlist(i);
    double* scores = new double[ql.size];  // float scores[ql.size];
    for (unsigned int j = 0, offset = samples->get_rloffsets(i); j < ql.size;)
      scores[j++] = eval_dp(featurematrix, offset++);
    //double *sortedlabels = copyextdouble_qsort(ql.labels, scores, ql.size);
    std::unique_ptr<double[]> sortedlabels = copyextdouble_mergesort<double,
        double>(ql.labels, scores, ql.size);
    score += scorer->evaluate_result_list(
        ResultList(ql.size, sortedlabels.get(), ql.qid));
    // delete[] sortedlabels;
    delete[] scores;
  }
  return nrankedlists ? score / nrankedlists : 0.0f;
}

void LTR_Algorithm::score_dataset(quickrank::data::Dataset &dataset,
                                  quickrank::Score* scores) const {
  if (dataset.format() != quickrank::data::Dataset::VERT)
    dataset.transpose();
  for (unsigned int q = 0; q < dataset.num_queries(); q++) {
    std::shared_ptr<quickrank::data::QueryResults> r = dataset.getQueryResults(
        q);
    score_query_results(r, scores, dataset.num_instances());
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
