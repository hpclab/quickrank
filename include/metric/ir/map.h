/*
 * QuickRank - A C++ suite of Learning to Rank algorithms
 * Webpage: http://quickrank.isti.cnr.it/
 * Contact: quickrank@isti.cnr.it
 *
 * Unless explicitly acquired and licensed from Licensor under another
 * license, the contents of this file are subject to the Reciprocal Public
 * License ("RPL") Version 1.5, or subsequent versions as allowed by the RPL,
 * and You may not copy or use this file in either source code or executable
 * form, except in compliance with the terms and conditions of the RPL.
 *
 * All software distributed under the RPL is provided strictly on an "AS
 * IS" basis, WITHOUT WARRANTY OF ANY KIND, EITHER EXPRESS OR IMPLIED, AND
 * LICENSOR HEREBY DISCLAIMS ALL SUCH WARRANTIES, INCLUDING WITHOUT
 * LIMITATION, ANY WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
 * PURPOSE, QUIET ENJOYMENT, OR NON-INFRINGEMENT. See the RPL for specific
 * language governing rights and limitations under the RPL.
 *
 * Contributor:
 *   HPC. Laboratory - ISTI - CNR - http://hpc.isti.cnr.it/
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

  /// Returns the name of the metric.
  virtual std::string name() const {
    return NAME_;
  }

  static const std::string NAME_;

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
