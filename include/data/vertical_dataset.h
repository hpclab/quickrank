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
#pragma once

#include <iostream>
#include <memory>
#include <vector>

#include "types.h"
#include "data/queryresults.h"
#include "dataset.h"

namespace quickrank {
namespace data {

/**
 * This class implements a Dataset to be used for a L-t-R task.
 *
 * The internal representation is vertical: a row vector
 * of size \a num_instances() x \a num_features().
 * (A training instance is indeed a document.)
 * We allow to directly
 * access the internal representation through the function \a at()
 * to support fast access and custom high performance implementations.
 * Representation is vertical, i.e., a matrix features x documents.
 */
class VerticalDataset {
 public:

  /// Allocates a vertical dataset by copying and transposing an horizontal one.
  ///
  /// \param h_dataset The horizontal dataset.
  VerticalDataset(std::shared_ptr<Dataset> h_dataset);
  virtual ~VerticalDataset();

  /// Avoid inefficient copy constructor
  VerticalDataset(const VerticalDataset &other) = delete;
  /// Avoid inefficient copy assignment
  VerticalDataset &operator=(const VerticalDataset &) = delete;

  /// Returns a pointer to a specific data item.
  ///
  /// \param document_id The document of interest.
  /// \param feature_id The feature of interest.
  /// \returns A reference to the requested feature value of the given document id.
  quickrank::Feature *at(size_t document_id, size_t feature_id) {
    return data_ + document_id + feature_id * num_instances_;
  }

  /// Returns the value of the i-th relevance label.
  Label getLabel(size_t document_id) {
    return labels_[document_id];
  }

  /// Returns the offset in the internal data strcutures of the i-th query results list.
  ///
  /// \param i The i-th query results list of interest.
  /// \returns The offset of the first document in the i-th query results list.
  ///     This can be used to later invoke the \a at() function.
  unsigned int offset(size_t i) const {
    return offsets_[i];
  }

  /// Returns the i-th QueryResults in the dataset.
  ///
  /// \param i The i-th query results list of interest.
  /// \returns The requested QueryResults.
  std::unique_ptr<QueryResults> getQueryResults(size_t i) const;

  /// Returns the number of features used to represent a document.
  unsigned int num_features() const {
    return num_features_;
  }
  /// Returns the number of queries in the dataset.
  unsigned int num_queries() const {
    return num_queries_;
  }
  /// Returns the number of documents in the dataset.
  unsigned int num_instances() const {
    return num_instances_;
  }

 private:

  size_t num_features_;
  size_t num_queries_;
  size_t num_instances_;

  quickrank::Feature *data_ = NULL;
  quickrank::Label *labels_ = NULL;
  std::vector<size_t> offsets_;

  /// The output stream operator.
  /// Prints the data reading time stats
  friend std::ostream &operator<<(std::ostream &os, const VerticalDataset &me) {
    return me.put(os);
  }

  /// Prints the data reading time stats
  virtual std::ostream &put(std::ostream &os) const;

};

}  // namespace data
}  // namespace quickrank

