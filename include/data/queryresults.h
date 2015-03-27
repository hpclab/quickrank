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
