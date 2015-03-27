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

