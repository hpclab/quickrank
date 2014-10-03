#include "learning/ranker.h"

#ifdef _OPENMP
#include <omp.h>
#else
#include "utils/omp-stubs.h"
#endif

// NOTE this replaces a "lot" of methods used in lmart, ranker, evaluator
float LTR_Algorithm::compute_score(DataPointDataset *samples, Metric *scorer) {
  const unsigned int nrankedlists = samples->get_nrankedlists();
  unsigned int * const rloffsets = samples->get_rloffsets();
  float * const * const featurematrix = samples->get_fmatrix();
  float score = 0.0f;
#pragma omp parallel for reduction(+:score)
  for (unsigned int i = 0; i < nrankedlists; ++i) {
    qlist ql = samples->get_qlist(i);
    double* scores = new double[ql.size];  // float scores[ql.size];
    for (unsigned int j = 0, offset = rloffsets[i]; j < ql.size;)
      scores[j++] = eval_dp(featurematrix, offset++);
    double *sortedlabels = copyextdouble_qsort(ql.labels, scores, ql.size);
    score += scorer->compute_score(qlist(ql.size, sortedlabels, ql.qid));
    delete[] sortedlabels;
    delete[] scores;
  }
  return nrankedlists ? score / nrankedlists : 0.0f;
}
