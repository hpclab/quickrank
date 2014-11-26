#include "data/rankedresults.h"
#include "utils/mergesorter.h"

namespace quickrank {
namespace data {

RankedResults::RankedResults(std::shared_ptr<QueryResults> results,
                             Score* scores) {

  // unsigned int *idx = idxdouble_qsort(trainingmodelscores+offset, ql.size);
  num_results_ = results->num_results();
  unmap_ = idxdouble_mergesort<Score>(scores, num_results_);

  labels_ = new Label[num_results_];
  scores_ = new Score[num_results_];
  for (unsigned int i = 0; i < num_results_; i++) {
    labels_[i] = results->labels()[unmap_[i]];
    scores_[i] = scores[unmap_[i]];
  }
}

RankedResults::~RankedResults() {
  if (labels_)
    delete[] labels_;
  if (scores_)
    delete[] scores_;
  if (unmap_)
    delete[] unmap_;
}

}  // namespace data
}  // namespace quickrank

