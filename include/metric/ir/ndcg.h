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

#ifndef QUICKRANK_METRIC_IR_NDCG_H_
#define QUICKRANK_METRIC_IR_NDCG_H_

#include "types.h"
#include "dcg.h"

namespace quickrank {
namespace metric {
namespace ir {

/**
 * This class implements the Normalized Discounted cumulative Gain NDCG\@k measure.
 *
 * NDCG is measured as: \f$ NDCG_k = \frac{1}{IDCG_k}\sum_{i=1}^k \frac{2^{l_i}-1}{\log_2 (i+1)}\f$,
 * where \f$l_i\f$ is the relevance label of the i-th document,
 * and \f$IDCG_k\f$ is the NDCG\@K of a perfectly orderd result list.
 */
class Ndcg : public Dcg {
 public:
  explicit Ndcg(int k = NO_CUTOFF)
      : Dcg(k) {
  }
  virtual ~Ndcg() {
  }

  /// Returns the name of the metric.
  virtual std::string name() const {
    return NAME_;
  }

  static const std::string NAME_;

  /// \todo TODO: for only zero result slist Yahoo! LTR returns 0.5 instead of 0.0.
  ///             Make this choice available.
  /// \param rl A results list.
  /// \param scores a list of scores
  /// \return The quality score of the result list.
  virtual MetricScore evaluate_result_list(
      const quickrank::data::QueryResults* rl, const Score* scores) const;

  virtual std::unique_ptr<Jacobian> jacobian(
      std::shared_ptr<data::RankedResults> ranked) const;

 protected:
  /// Computes the IDCG\@K of a given list of labels.
  /// \param rl The given results list. Only labels are actually used.
  /// \return IDCG\@K for computed on the given labels.
  MetricScore compute_idcg(const quickrank::data::QueryResults* rl) const;

 private:
  friend std::ostream& operator<<(std::ostream& os, const Ndcg& ndcg) {
    return ndcg.put(os);
  }

  virtual std::ostream& put(std::ostream& os) const;

};

}  // namespace ir
}  // namespace metric
}  // namespace quickrank

#endif // QUICKRANK_NDCG_H_
