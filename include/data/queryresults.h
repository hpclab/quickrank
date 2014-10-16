#ifndef QUICKRANK_DATA_QUERYRESULTS_H_
#define QUICKRANK_DATA_QUERYRESULTS_H_

#include <boost/noncopyable.hpp>
#include <boost/container/vector.hpp>

#include "types.h"

namespace quickrank {
namespace data {

class QueryResults : private boost::noncopyable {
 public:

  /// Allocates an Query Results Object.
  ///
  /// This is used to store a the set of instances and their labels
  /// related to a specific query.
  /// \param n_instances The number of training instances (lines) in the dataset.
  /// \param n_features The number of features.
  QueryResults(unsigned int n_results, qr::Label* new_labels, qr::Feature* new_features);
  virtual ~QueryResults();



  qr::Feature* features() const {return features_;}
  qr::Label* labels() const {return labels_;}
  unsigned int num_results() const {return num_results_;}

 private:
  qr::Label* labels_;
  qr::Feature* features_;
  unsigned int num_results_;
};

} // namespace data
} // namespace quickrank


#endif
