/*
 * QuickRank - A C++ suite of Learning to Rank algorithms
 * Webpage: http://quickrank.isti.cnr.it/
 * Contact: quickrank@isti.cnr.it
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 *
 * Contributor:
 *   HPC. Laboratory - ISTI - CNR - http://hpc.isti.cnr.it/
 */
#include "data/queryresults.h"

namespace quickrank {
namespace data {

QueryResults::QueryResults(unsigned int n_results, quickrank::Label* new_labels,
                           quickrank::Feature* new_features) {
  num_results_ = n_results;
  labels_ = new_labels;
  features_ = new_features;
}

QueryResults::~QueryResults() {
}

struct external_sort_op_t {
  const Score* values_;
  external_sort_op_t(const Score* values) {values_=values;}
  bool operator() (int i,int j) {return (values_[i]>values_[j]);}
};

void QueryResults::indexing_of_sorted_labels(const Score* scores, unsigned int* dest) const {
  external_sort_op_t comp(scores);
  for (unsigned int i=0; i<num_results_; ++i)
    dest[i] = i;
  std::sort(dest, dest+num_results_, comp);
}

void QueryResults::sorted_labels(const Score* scores, Label* dest, const unsigned int cutoff) const {
  unsigned int* idx = new unsigned int [num_results_];
  indexing_of_sorted_labels(scores, idx);
  for (unsigned int i=0; i<num_results_ && i<cutoff; ++i)
    dest[i] = labels_[idx[i]];
  delete [] idx;
}

}  // namespace data
}  // namespace quickrank

