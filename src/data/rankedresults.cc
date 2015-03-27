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
#include "data/rankedresults.h"

namespace quickrank {
namespace data {

RankedResults::RankedResults(std::shared_ptr<QueryResults> results,
                             Score* scores) {

  num_results_ = results->num_results();
  unmap_ = new unsigned int [num_results_];
  results->indexing_of_sorted_labels(scores, unmap_);

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

