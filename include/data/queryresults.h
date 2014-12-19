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
  QueryResults(unsigned int n_results, Label* new_labels,
               Feature* new_features);
  virtual ~QueryResults();

  Feature* features() const {
    return features_;
  }
  Label* labels() const {
    return labels_;
  }
  unsigned int num_results() const {
    return num_results_;
  }

  /// Sorts the element of the current result list
  /// in descending order of the given \a scores vector
  /// and stores in \a dest the positions of the sorted labels.
  ///
  /// \param scores vector of scores used for reverse sorting.
  /// \param dest output of the sorting indexing.
  void indexing_of_sorted_labels(const Score* scores, unsigned int* dest) const;

  /// Sorts the element of the current result list
  /// in descending order of the given \a scores vector
  /// and stores the resulting sorted labels in \a dest.
  ///
  /// \param scores vector of scores used for reverse sorting.
  /// \param dest output of the labels sorting.
  /// \param cutoff number of labels of interest, i.e., length of \a dest.
  void sorted_labels(const Score* scores, Label* dest, const unsigned int cutoff) const;

 private:
  Label* labels_;
  Feature* features_;
  unsigned int num_results_;
};

}  // namespace data
}  // namespace quickrank

#endif
