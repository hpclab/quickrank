#include "data/dataset.h"

namespace quickrank {
namespace data {

Dataset::Dataset(unsigned int n_instances, unsigned int n_features){
  max_instances_ = n_instances;
  num_features_ = n_features;
  num_instances_ = 0;
  num_queries_ = 0;
  last_instance_id_ = 0;

  data_ = new qr::Feature [max_instances_*num_features_] ();  // 0 initialization
  labels_ = new qr::Label [max_instances_];               // no initialization

  offsets_.push_back(0);
}

Dataset::~Dataset() {
  delete [] data_;
  delete [] labels_;
}


void Dataset::addInstance(qr::QueryID q_id, qr::Label i_label,
                          boost::container::vector<qr::Feature> i_features) {

  if (i_features.size()>num_features_ || num_instances_==max_instances_) {
    std::cerr << "!!! Impossible to add a new instance to the dataset." << std::endl;
    exit(EXIT_FAILURE);
  }

  // update label and features
  labels_[num_instances_] = i_label;
  qr::Feature* new_instance = data_ + (num_instances_*num_features_);
  for (unsigned int i=0; i<i_features.size(); i++)
    new_instance[i] = i_features[i];

  // update offset of last query result
  if (num_instances_==0 || last_instance_id_!=q_id) {
    num_queries_++;
    offsets_.push_back(0);
    last_instance_id_ = q_id;
  }
  num_instances_++;
  offsets_.back() = num_instances_;
}

std::unique_ptr<QueryResults> Dataset::getQueryResults(unsigned int i) const {
  unsigned int num_results = offsets_[i+1]-offsets_[i];
  qr::Feature* start_data = data_ + (offsets_[i]*num_features_);
  qr::Label* start_label  = labels_ + offsets_[i];

  QueryResults* qr = new QueryResults(num_results, start_label, start_data);

  return std::unique_ptr<QueryResults>(qr);
}


} // namespace data
} // namespace quickrank
