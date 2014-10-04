/*
 * dcg.h
 *
 *  Created on: Oct 3, 2014
 *      Author: claudio
 */

#ifndef QUICKRANK_DCG_H_
#define QUICKRANK_DCG_H_

#include "types.h"
#include "metric.h"

namespace qr {
namespace metric {
namespace ir {

class Dcg : public Metric {
 public:
  explicit Dcg(int k = NO_CUTOFF) : Metric(k) {}
  virtual ~Dcg() {};

  virtual MetricScore evaluate_result_list(const qlist&) const;

  virtual Jacobian* get_jacobian(const qlist &ql) const;

 protected:
  double compute_dcg(double const*, const unsigned int, const unsigned int) const;

 private:
  Dcg(const Dcg&);
  Dcg& operator=(const Dcg&);


  friend std::ostream& operator<<(std::ostream& os, const Dcg& ndcg) {
    ndcg.print(os); return os; }
  virtual void print(std::ostream& os) const;

};

} // namespace ir
} // namespace metric
} // namespace qr

#endif // QUICKRANK_NDCG_H_
