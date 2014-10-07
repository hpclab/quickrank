/*
 * ndcg.h
 *
 *  Created on: Oct 3, 2014
 *      Author: claudio
 */

#ifndef QUICKRANK_METRIC_IR_NDCG_H_
#define QUICKRANK_METRIC_IR_NDCG_H_

#include "types.h"
#include "dcg.h"

namespace qr {
namespace metric {
namespace ir {

class Ndcg : public Dcg {
 public:
  explicit Ndcg(int k = NO_CUTOFF) : Dcg(k) {}
  virtual ~Ndcg() {};

  virtual MetricScore evaluate_result_list(const ResultList&) const;

  virtual std::unique_ptr<Jacobian> get_jacobian(const ResultList &ql) const;

 protected:
  double compute_idcg(double const*, const unsigned int, const unsigned int) const;

 private:
  friend std::ostream& operator<<(std::ostream& os, const Ndcg& ndcg) {
    ndcg.print(os); return os;
  }

  virtual void print(std::ostream& os) const;

};

} // namespace ir
} // namespace metric
} // namespace qr

#endif // QUICKRANK_NDCG_H_
