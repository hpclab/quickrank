#ifndef QUICKRANK_METRIC_IR_METRIC_H_
#define QUICKRANK_METRIC_IR_METRIC_H_

#include <iostream>
#include <climits>
#include <memory>
#include <boost/noncopyable.hpp>

#include "learning/dpset.h"

#include "types.h"


namespace qr {
namespace metric {
namespace ir {

/**
 * This class implements the basic functionalities of an IR evaluation metric.
 */
class Metric : private boost::noncopyable
{
 public:
  /// This should be used when no cut-off on the results list is required.
  static const unsigned int NO_CUTOFF = UINT_MAX;

  // TODO: Fix k = 0, no cutoff
  /// Creates a new metric with the specified cut-off threshold.
  ///
  /// \param k The cut-off threshold.
  explicit Metric(unsigned int k = NO_CUTOFF) { set_cutoff(k); }
  virtual ~Metric() {};

  /// Returns the current cut-off of the Metric.
  unsigned int cutoff() const { return cutoff_; }
  /// Updates the cut-off of the Metric.
  void set_cutoff(unsigned int k) { cutoff_ = k == 0 ? NO_CUTOFF : k; }

  /// Measures the quality of the given results list according to the Metric.
  ///
  /// \param rl A results list.
  /// \return The quality score of the result list.
  virtual MetricScore evaluate_result_list(const ResultList& rl) const = 0;

  /// Computes the Jacobian matrix.
  /// This is a symmetric matrix storing the metric change when two documents scores
  /// are swaped.
  /// \param rl A results list.
  /// \return A smart-pointer to the Jacobian Matrix.
  /// \todo TODO: provide def implementation
  virtual std::unique_ptr<Jacobian> get_jacobian(const ResultList &rl) const { return std::unique_ptr<Jacobian>(); }

 private:

  /// The metric cutoff.
  unsigned int cutoff_;

  /// The output stream operator.
  // TODO: check this together
  friend std::ostream& operator<<(std::ostream& os, const Metric& m) {
    m.print(os); return os;
  }
  /// Prints the shortname of the Metric, e.g., "NDCG@K"
  virtual void print(std::ostream& os) const {os << "Empty";}

};

} // namespace ir
} // namespace metric
} // namespace qr


#endif
