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
 * see: McSherry, Frank, and Marc Najork. "Computing information retrieval
 * performance measures efficiently in the presence of tied scores."
 * In Advances in information retrieval, pp. 414-421. Springer Berlin Heidelberg, 2008.
 */
class Tndcg : public Ndcg {
 public:
  explicit Tndcg(int k = NO_CUTOFF)
      : Ndcg(k) {
  }
  virtual ~Tndcg() {
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
  /// Computes the TNDCG\@K of a given list of labels.
  /// \param rl The given results list. Only labels are actually used.
  /// \param scores The scores to be used to re-order the result list.
  /// \return TNDCG\@K for computed on the given labels.
  MetricScore compute_tndcg(const quickrank::data::QueryResults* rl,
                            const Score* scores) const;

 private:
  friend std::ostream& operator<<(std::ostream& os, const Tndcg& tndcg) {
    return tndcg.put(os);
  }

  virtual std::ostream& put(std::ostream& os) const;

};

}  // namespace ir
}  // namespace metric
}  // namespace quickrank

#endif // QUICKRANK_METRIC_IR_TNDCG_H_
