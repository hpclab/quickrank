#ifndef QUICKRANK_DATA_DATASET_H_
#define QUICKRANK_DATA_DATASET_H_

#include <boost/container/vector.hpp>

// TODO: rename to ltrdata.h

#include "types.h"
#include "data/queryresults.h"

namespace quickrank {
namespace data {

class Dataset {
 public:

  /// Allocates an empty Dataset of given size.
  ///
  /// \param n_instances The number of training instances (lines) in the dataset.
  /// \param n_features The number of features.
  Dataset(unsigned int n_instances, unsigned int n_features);
  virtual ~Dataset();


  // TODO: begin, end
  QueryResults getQueryResults(unsigned int i);

  void addInstance(qr::QueryID q_id, qr::Label i_label, boost::container::vector<qr::Feature> i_features);

  unsigned int num_features()  {return num_features_;}
  unsigned int num_queries()   {return num_queries_;}
  unsigned int num_instances() {return num_instances_;}

  // - support normalization
  // - support discretisation, or simply provide discr.ed thresholds
  // - support horiz. and vert. sampling

 private:
  unsigned int num_features_;
  unsigned int num_queries_;
  unsigned int num_instances_;

  qr::Feature* data_;
  qr::Label* labels_;
  boost::container::vector<unsigned int> offsets_;

  unsigned int last_instance_id_;
  unsigned int max_instances_;
};

} // namespace data
} // namespace quickrank


#endif
