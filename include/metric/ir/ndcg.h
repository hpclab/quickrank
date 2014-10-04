/*
 * ndcg.h
 *
 *  Created on: Oct 3, 2014
 *      Author: claudio
 */

#ifndef QUICKRANK_NDCG_H_
#define QUICKRANK_NDCG_H_

#include "types.h"
#include "metric.h"

namespace qr {
namespace metric {
namespace ir {

class Ndcg : public Metric {
 public:
  explicit Ndcg(int k = NO_CUTOFF) : Metric(k) {}
  virtual ~Ndcg() {};

  virtual MetricScore evaluate_result_list(const qlist&) const;

  virtual Jacobian* get_jacobian(const qlist &ql) const;

 protected:
  double compute_idcg(double const*, const unsigned int, const unsigned int) const;
  double compute_dcg(double const*, const unsigned int, const unsigned int) const;

 private:
  Ndcg(const Ndcg&);
  Ndcg& operator=(const Ndcg&);

  friend std::ostream& operator<<(std::ostream& os, const Ndcg& ndcg) {
    ndcg.print(os); return os; }
  virtual void print(std::ostream& os) const;

};

} // namespace ir
} // namespace metric
} // namespace qr

#endif // QUICKRANK_NDCG_H_
