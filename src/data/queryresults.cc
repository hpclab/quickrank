#include "data/queryresults.h"

namespace quickrank {
namespace data {

QueryResults::QueryResults(unsigned int n_results, quickrank::Label* new_labels,
                           quickrank::Feature* new_features) {
  num_results_ = n_results;
  labels_ = new_labels;
  features_ = new_features;
}

QueryResults::~QueryResults() {
}

}  // namespace data
}  // namespace quickrank

