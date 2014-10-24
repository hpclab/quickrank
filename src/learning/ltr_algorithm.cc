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

}  // namespace learning
}  // namespace quickrank
