#ifndef QUICKRANK_DATA_DATASET_H_
#define QUICKRANK_DATA_DATASET_H_

#include <boost/noncopyable.hpp>
#include <boost/container/vector.hpp>

// TODO: rename to ltrdata.h

#include "types.h"
#include "data/queryresults.h"

namespace quickrank {
namespace data {

class Dataset : private boost::noncopyable {
 public:

  enum Format {HORIZ, VERT};

  /// Allocates an empty Dataset of given size.
  ///
  /// \param n_instances The number of training instances (lines) in the dataset.
  /// \param n_features The number of features.
  Dataset(unsigned int n_instances, unsigned int n_features);
  virtual ~Dataset();


  qr::Feature* at(unsigned int document_id, unsigned int feature_id) {
    return (format_==HORIZ) ? (data_ + document_id*num_features_ + feature_id)
        : (data_ + document_id + feature_id*num_instances_); }

  unsigned int offset(unsigned int i) const {return offsets_[i];}

  // TODO: add an iterator
  std::unique_ptr<QueryResults> getQueryResults(unsigned int i) const;

  void addInstance(qr::QueryID q_id, qr::Label i_label, boost::container::vector<qr::Feature> i_features);

  unsigned int num_features() const   {return num_features_;}
  unsigned int num_queries() const    {return num_queries_;}
  unsigned int num_instances() const  {return num_instances_;}

  Format format() const {return format_;}

  void transpose();

  // - support normalization
  // - support discretisation, or simply provide discr.ed thresholds
  // - support horiz. and vert. sampling

 private:

  unsigned int num_features_;
  unsigned int num_queries_;
  unsigned int num_instances_;

  Format format_;

  qr::Feature* data_;
  qr::Label* labels_;
  boost::container::vector<unsigned int> offsets_;

  unsigned int last_instance_id_;
  unsigned int max_instances_;
};

} // namespace data
} // namespace quickrank


#endif
