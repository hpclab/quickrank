#ifndef QUICKRANK_METRIC_IR_METRIC_H_
#define QUICKRANK_METRIC_IR_METRIC_H_

#include <iostream>
#include <climits>
#include <memory>
#include <boost/noncopyable.hpp>

#include "data/queryresults.h"
#include "data/dataset.h"
#include "learning/dpset.h"
#include "types.h"

namespace quickrank {
namespace metric {
namespace ir {

/**
 * This class implements the basic functionalities of an IR evaluation metric.
 */
class Metric : private boost::noncopyable {
 public:
  /// This should be used when no cut-off on the results list is required.
  static const unsigned int NO_CUTOFF = UINT_MAX;

  // TODO: Fix k = 0, no cutoff
  /// Creates a new metric with the specified cut-off threshold.
  ///
  /// \param k The cut-off threshold.
  explicit Metric(unsigned int k = NO_CUTOFF) {
    set_cutoff(k);
  }
  virtual ~Metric() {
  }
  ;

  /// Returns the current cut-off of the Metric.
  unsigned int cutoff() const {
    return cutoff_;
  }
  /// Updates the cut-off of the Metric.
  void set_cutoff(unsigned int k) {
    cutoff_ = k == 0 ? NO_CUTOFF : k;
  }

  /// Measures the quality of the given results list according to the Metric.
  ///
  /// \param rl A results list.
  /// \return The quality score of the result list.
  virtual MetricScore evaluate_result_list(const ResultList& rl) const = 0;

  /// Measures the quality of the given results list according to the Metric.
  ///
  /// \param rl A results list.
  /// \param scores a list of scores
  /// \return The quality score of the result list.
  virtual MetricScore evaluate_result_list(
      const quickrank::data::QueryResults* rl, const Score* scores) const = 0;
  virtual MetricScore evaluate_dataset(const quickrank::data::Dataset &dataset,
                                       const Score* scores) const {
    if (dataset.num_queries() == 0)
      return 0.0;
    MetricScore avg_score = 0.0;
    for (unsigned int q = 0; q < dataset.num_queries(); q++) {
      std::shared_ptr<quickrank::data::QueryResults> r =
          dataset.getQueryResults(q);
      avg_score += evaluate_result_list(r.get(), scores);
      scores += r->num_results();
    }
    avg_score /= (MetricScore) dataset.num_queries();
    return avg_score;
  }

  /// Computes the Jacobian matrix.
  /// This is a symmetric matrix storing the metric change when two documents scores
  /// are swaped.
  /// \param rl A results list.
  /// \return A smart-pointer to the Jacobian Matrix.
  /// \todo TODO: provide def implementation
  virtual std::unique_ptr<Jacobian> get_jacobian(const ResultList &rl) const {
    return std::unique_ptr<Jacobian>();
  }

 private:

  /// The metric cutoff.
  unsigned int cutoff_;

  /// The output stream operator.
  friend std::ostream& operator<<(std::ostream& os, const Metric& m) {
    return m.put(os);
  }
  /// Prints the short name of the Metric, e.g., "NDCG@K"
  virtual std::ostream& put(std::ostream& os) const = 0;

};

}  // namespace ir
}  // namespace metric
}  // namespace quickrank

#endif
