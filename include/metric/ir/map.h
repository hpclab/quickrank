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

namespace quickrank {
namespace metric {
namespace ir {

/**
 * This class implements the average precision AP\@k measure.
 *
 * \todo TODO: test correctness
 */
class Map : public Metric {
 public:
  explicit Map(int k = NO_CUTOFF)
      : Metric(k) {
  }
  virtual ~Map() {
  }
  ;

  virtual MetricScore evaluate_result_list(
      const quickrank::data::QueryResults* rl, const Score* scores) const;

  virtual std::unique_ptr<Jacobian> jacobian(
      std::shared_ptr<data::RankedResults> ranked) const;

 protected:

 private:
  friend std::ostream& operator<<(std::ostream& os, const Map& map) {
    return map.put(os);
  }

  virtual std::ostream& put(std::ostream& os) const;

};

}  // namespace ir
}  // namespace metric
}  // namespace quickrank

#endif // QUICKRANK_MAP_H_
