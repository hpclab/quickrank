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
#include "data/dataset.h"

#include <iomanip>
#include <cstring>

namespace quickrank {
namespace data {

Dataset::Dataset(size_t n_instances, size_t n_features) {
  max_instances_ = n_instances;
  num_features_ = n_features;
  num_instances_ = 0;
  num_queries_ = 0;
  last_instance_id_ = 0;

  if (posix_memalign((void **) &data_, 16,
                     max_instances_ * num_features_ * sizeof(Feature)) != 0) {
    std::cerr << "!!! Impossible to allocate memory for dataset storage."
              << std::endl;
    exit(EXIT_FAILURE);
  }
  std::memset(data_, 0, max_instances_ * num_features_ * sizeof(Feature));

  if (posix_memalign((void **) &labels_, 16, max_instances_ * sizeof(Label))
      != 0) {
    std::cerr
        << "!!! Impossible to allocate memory for relevance labels storage."
        << std::endl;
    exit(EXIT_FAILURE);
  }

  offsets_.push_back(0);
}

Dataset::~Dataset() {
  if (data_)
    free(data_);
  if (labels_)
    free(labels_);
}

void Dataset::addInstance(QueryID q_id, Label i_label,
                          std::vector<Feature> i_features) {

  if (i_features.size() > num_features_ || num_instances_ == max_instances_) {
    std::cerr << "!!! Impossible to add a new instance to the dataset."
              << std::endl;
    exit(EXIT_FAILURE);
  }

  // update label and features
  labels_[num_instances_] = i_label;
  quickrank::Feature *new_instance = data_ + (num_instances_ * num_features_);
  for (size_t i = 0; i < i_features.size(); i++)
    new_instance[i] = i_features[i];

  // update offset of last query result
  if (num_instances_ == 0 || last_instance_id_ != q_id) {
    num_queries_++;
    offsets_.push_back(0);
    last_instance_id_ = q_id;
  }
  num_instances_++;
  offsets_.back() = num_instances_;
}

std::unique_ptr<QueryResults> Dataset::getQueryResults(size_t i) const {
  size_t num_results = offsets_[i + 1] - offsets_[i];
  quickrank::Feature *start_data = data_ + offsets_[i] * num_features_;
  quickrank::Label *start_label = labels_ + offsets_[i];

  QueryResults *qr = new QueryResults(num_results, start_label, start_data);

  return std::unique_ptr<QueryResults>(qr);
}

std::ostream &Dataset::put(std::ostream &os) const {
  os << "#\t Dataset size: " << num_instances_ << " x " << num_features_
     << " (instances x features)" << std::endl << "#\t Num queries: "
     << num_queries_ << " | Avg. len: " << std::setprecision(3)
     << num_instances_ / (float) num_queries_ << std::endl;
  return os;
}

}  // namespace data
}  // namespace quickrank
