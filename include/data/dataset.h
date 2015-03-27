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
#ifndef QUICKRANK_DATA_DATASET_H_
#define QUICKRANK_DATA_DATASET_H_

#include <iostream>
#include <boost/noncopyable.hpp>
#include <boost/container/vector.hpp>

// TODO: rename to ltrdata.h

#include "types.h"
#include "data/queryresults.h"

namespace quickrank {
namespace data {

/**
 * This class implements a Dataset to be used for a L-t-R task.
 *
 * The internal representation is quite simple: a row vector
 * of size \a num_instances() x \a num_features().
 * (A training instance is indeed a document.)
 * We allow to directly
 * access the internal representation throuhh the functinon \a at()
 * to support fast access and custom high performance implementations.
 * Both horizontal (instances x features) and vertical (features x instances)
 * representations are supported.
 */
class Dataset : private boost::noncopyable {
 public:

  enum Format {
    HORIZ,
    VERT
  };

  /// Allocates an empty Dataset of given sizein horizontal format.
  ///
  /// \param n_instances The number of training instances (lines) in the dataset.
  /// \param n_features The number of features.
  Dataset(unsigned int n_instances, unsigned int n_features);
  virtual ~Dataset();

  /// Returns a pointer to a specific data item.
  ///
  /// \param document_id The document of interest.
  /// \param feature_id The feature of interest.
  /// \returns A reference to the requested feature value of the given document id.
  quickrank::Feature* at(unsigned int document_id, unsigned int feature_id) {
    return
        (format_ == HORIZ) ?
            (data_ + document_id * num_features_ + feature_id) :
            (data_ + document_id + feature_id * num_instances_);
  }

  /// Returns the value of the i-th relevance label.
  Label getLabel(unsigned int document_id) {
    return labels_[document_id];
  }

  /// Returns the offset in the internal data strcutures of the i-th query results list.
  ///
  /// \param i The i-th query results list of interest.
  /// \returns The offset of the first document in the i-th query results list.
  ///     This can be used to later invoke the \a at() function.
  unsigned int offset(unsigned int i) const {
    return offsets_[i];
  }

  /// Returns the i-th QueryResults in the dataset.
  ///
  /// \param i The i-th query results list of interest.
  /// \returns The requested QueryResults.
  std::unique_ptr<QueryResults> getQueryResults(unsigned int i) const;

  /// Add a new training instance, i.e., a labeled document, to the dataset.
  ///
  /// \warning Currently the addition words only when data is in HORIZ format.
  /// \param q_id The query ID.
  /// \param i_label The relevance label of the result.
  /// \param i_features The feature vector of the document.
  void addInstance(QueryID q_id, Label i_label,
                   boost::container::vector<Feature> i_features);

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
  /// Returns current format, HORIZ vs. VERT, of the dataset.
  Format format() const {
    return format_;
  }

  /// Transposes the matrix.
  ///
  /// The internal representation is transformed from HORIZ to VERT
  /// or viceversa.
  void transpose();

  // - support normalization
  // - support discretisation, or simply provide discr.ed thresholds
  // - support horiz. and vert. sampling

 private:

  unsigned int num_features_;
  unsigned int num_queries_;
  unsigned int num_instances_;

  Format format_;

  quickrank::Feature* data_ = NULL;
  quickrank::Label* labels_ = NULL;
  boost::container::vector<unsigned int> offsets_;

  unsigned int last_instance_id_;
  unsigned int max_instances_;

  /// The output stream operator.
  /// Prints the data reading time stats
  friend std::ostream& operator<<(std::ostream& os, const Dataset& me) {
    return me.put(os);
  }

  /// Prints the data reading time stats
  virtual std::ostream& put(std::ostream& os) const;

};

}  // namespace data
}  // namespace quickrank

#endif
