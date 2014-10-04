/*
 * map.h
 *
 *  Created on: Oct 3, 2014
 *      Author: claudio
 */

#ifndef QUICKRANK_METRIC_IR_MAP_H_
#define QUICKRANK_METRIC_IR_MAP_H_

#include "types.h"
#include "metric.h"

namespace qr {
namespace metric {
namespace ir {

// TODO: test correctness
class Map : public Metric {
 public:
  explicit Map(int k = NO_CUTOFF) : Metric(k) {}
  virtual ~Map() {};

  virtual MetricScore evaluate_result_list(const ResultList&) const;

  virtual Jacobian* get_jacobian(const ResultList &ql) const;

 protected:

 private:
  Map(const Map&);
  Map& operator=(const Map&);


  friend std::ostream& operator<<(std::ostream& os, const Map& map) {
    map.print(os); return os; }
  virtual void print(std::ostream& os) const;

};

} // namespace ir
} // namespace metric
} // namespace qr

#endif // QUICKRANK_NDCG_H_
