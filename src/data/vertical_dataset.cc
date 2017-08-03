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
#include "data/vertical_dataset.h"

#include <iomanip>

namespace quickrank {
namespace data {

VerticalDataset::VerticalDataset(std::shared_ptr<Dataset> h_dataset) {
  num_features_ = h_dataset->num_features();
  num_instances_ = h_dataset->num_instances();
  num_queries_ = h_dataset->num_queries();

  // transpose dataset
  if (posix_memalign((void **) &data_,
                     16,
                     num_instances_ * num_features_ * sizeof(Feature)) != 0) {
    std::cerr
        << "!!! Impossible to allocate memory for transposed dataset storage."
        << std::endl;
    exit(EXIT_FAILURE);
  }

  quickrank::Feature *h_data = h_dataset->at(0, 0);
  #pragma omp parallel for
  for (size_t i = 0; i < num_instances_; ++i) {
    for (size_t f = 0; f < num_features_; ++f) {
      data_[f * num_instances_ + i] = h_data[i * num_features_ + f];
    }
  }

  // allocate labels
  if (posix_memalign((void **) &labels_, 16, num_instances_ * sizeof(Label))
      != 0) {
    std::cerr
        << "!!! Impossible to allocate memory for relevance labels storage."
        << std::endl;
    exit(EXIT_FAILURE);
  }

  #pragma omp parallel for
  for (size_t i = 0; i < num_instances_; ++i)
    labels_[i] = h_dataset->getLabel(i);

  offsets_.resize(num_queries_ + 1);

  #pragma omp parallel for
  for (size_t i = 0; i < num_queries_ + 1; ++i)
    offsets_[i] = h_dataset->offset(i);
}

VerticalDataset::~VerticalDataset() {
  if (data_)
    free(data_);
  if (labels_)
    free(labels_);
}


std::unique_ptr<QueryResults> VerticalDataset::getQueryResults(size_t i) const {
  size_t num_results = offsets_[i + 1] - offsets_[i];
  quickrank::Feature *start_data = data_ + offsets_[i];
  quickrank::Label *start_label = labels_ + offsets_[i];

  QueryResults *qr = new QueryResults(num_results, start_label, start_data);

  return std::unique_ptr<QueryResults>(qr);
}


std::ostream &VerticalDataset::put(std::ostream &os) const {
  os << "#\t Vertical Dataset size: " << num_instances_ << " x "
     << num_features_
     << " (instances x features)" << std::endl << "#\t Num queries: "
     << num_queries_ << " | Avg. len: " << std::setprecision(3)
     << num_instances_ / (float) num_queries_ << std::endl;
  return os;
}

}  // namespace data
}  // namespace quickrank
