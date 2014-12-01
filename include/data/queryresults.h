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
#ifndef QUICKRANK_DATA_QUERYRESULTS_H_
#define QUICKRANK_DATA_QUERYRESULTS_H_

#include <boost/noncopyable.hpp>
#include <boost/container/vector.hpp>

#include "types.h"

namespace quickrank {
namespace data {

/// This class wraps a set of results for a given query.
///
/// The internal data representation is the same as the
/// \a Dataset it comes from.
/// \todo TODO: it seems we need also a class withouth features
class QueryResults : private boost::noncopyable {
 public:

  /// Allocates an Query Results Object.
  ///
  /// This is used to store a the set of instances and their labels
  /// related to a specific query.
  /// \param n_instances The number of training instances (lines) in the dataset.
  /// \param n_features The number of features.
  QueryResults(unsigned int n_results, quickrank::Label* new_labels,
               quickrank::Feature* new_features);
  virtual ~QueryResults();

  // get i,j?
  // get doc-vector
  // get fx-vector

  quickrank::Feature* features() const {
    return features_;
  }
  quickrank::Label* labels() const {
    return labels_;
  }
  unsigned int num_results() const {
    return num_results_;
  }

 private:
  quickrank::Label* labels_;
  quickrank::Feature* features_;
  unsigned int num_results_;
};

}  // namespace data
}  // namespace quickrank

#endif
