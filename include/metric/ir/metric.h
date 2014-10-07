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

class Metric : private boost::noncopyable
{
 public:
  static const unsigned int NO_CUTOFF = UINT_MAX - 1;

  // TODO: Fix k = 0, no cutoff
  explicit Metric(unsigned int k = NO_CUTOFF) { set_cutoff(k); }

  virtual ~Metric() {};

  unsigned int cutoff() const { return cutoff_; }
  void set_cutoff(unsigned int k) { cutoff_ = k >= NO_CUTOFF ? NO_CUTOFF : k; }

  virtual MetricScore evaluate_result_list(const ResultList&) const = 0;

  // TODO: provide def implementation
  virtual std::unique_ptr<Jacobian> get_jacobian(const ResultList &ql) const { return std::unique_ptr<Jacobian>(); }

 private:
  unsigned int cutoff_;

  // TODO: check this together
  friend std::ostream& operator<<(std::ostream& os, const Metric& m) {
    m.print(os); return os;
  }

  virtual void print(std::ostream& os) const {os << "Empty";}

};

} // namespace ir
} // namespace metric
} // namespace qr


#endif
