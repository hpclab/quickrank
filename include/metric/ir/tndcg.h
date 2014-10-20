/*
 * tndcg.h
 *
 *  Created on: Oct 3, 2014
 *      Author: claudio
 */

#ifndef QUICKRANK_METRIC_IR_TNDCG_H_
#define QUICKRANK_METRIC_IR_TNDCG_H_

#include "types.h"
#include "ndcg.h"

namespace quickrank {
namespace metric {
namespace ir {

/**
 * This class implements a Tie-aware version of Normalized Discounted Cumulative Gain TNDCG\@k measure.
 *
 * \todo TODO: to implement this, predicted score must be available to find ties!
 *
 * see: McSherry, Frank, and Marc Najork. "Computing information retrieval
 * performance measures efficiently in the presence of tied scores."
 * In Advances in information retrieval, pp. 414-421. Springer Berlin Heidelberg, 2008.
 */
class Tndcg : public Ndcg {
 public:
  explicit Tndcg(int k = NO_CUTOFF) : Ndcg(k) {}
  virtual ~Tndcg() {};

 private:
  friend std::ostream& operator<<(std::ostream& os, const Tndcg& tndcg) {
    return tndcg.put(os);
  }

  virtual std::ostream& put(std::ostream& os) const;

};

} // namespace ir
} // namespace metric
} // namespace quickrank

#endif // QUICKRANK_METRIC_IR_TNDCG_H_
