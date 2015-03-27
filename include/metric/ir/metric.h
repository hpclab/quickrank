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
#ifndef QUICKRANK_METRIC_IR_METRIC_H_
#define QUICKRANK_METRIC_IR_METRIC_H_

#include <iostream>
#include <climits>
#include <memory>
#include <boost/noncopyable.hpp>

#include "data/queryresults.h"
#include "data/rankedresults.h"
#include "data/dataset.h"
#include "types.h"

namespace quickrank {
namespace metric {
namespace ir {

/**
 * This class implements the basic functionalities of an IR evaluation metric.
 */
class Metric : private boost::noncopyable {
 public:
  /// This should be used when no cut-off on the results list is required.
  static const unsigned int NO_CUTOFF = UINT_MAX;

  /// Creates a new metric with the specified cut-off threshold.
  ///
  /// \param k The cut-off threshold.
  explicit Metric(unsigned int k = NO_CUTOFF) {
    set_cutoff(k);
  }
  virtual ~Metric() {
  }

  /// Returns the name of the metric.
  virtual std::string name() const = 0;

  /// Returns the current cut-off of the Metric.
  unsigned int cutoff() const {
    return cutoff_;
  }
  /// Updates the cut-off of the Metric.
  void set_cutoff(unsigned int k) {
    cutoff_ = k == 0 ? NO_CUTOFF : k;
  }

  /// Measures the quality of the given results list according to the Metric.
  ///
  /// \param rl A results list.
  /// \param scores a list of scores
  /// \return The quality score of the result list.
  virtual MetricScore evaluate_result_list(
      const quickrank::data::QueryResults* rl, const Score* scores) const = 0;
  virtual MetricScore evaluate_dataset(
      const std::shared_ptr<data::Dataset> dataset, const Score* scores) const {
    if (dataset->num_queries() == 0)
      return 0.0;
    MetricScore avg_score = 0.0;
    for (unsigned int q = 0; q < dataset->num_queries(); q++) {
      std::shared_ptr<data::QueryResults> r = dataset->getQueryResults(q);
      avg_score += evaluate_result_list(r.get(), scores);
      scores += r->num_results();
    }
    avg_score /= (MetricScore) dataset->num_queries();
    return avg_score;
  }

  /// Computes the Jacobian matrix.
  /// This is a symmetric matrix storing the metric "decrease" when two documents scores
  /// are swapped.
  /// \param rl A results list.
  /// \return A smart-pointer to the Jacobian Matrix.
  /// \todo TODO: provide def implementation
  virtual std::unique_ptr<Jacobian> jacobian(
      std::shared_ptr<data::RankedResults> ranked) const {
    auto jacobian = std::unique_ptr<Jacobian>(
        new Jacobian(ranked->num_results()));
    auto results = std::shared_ptr<data::QueryResults>(
        new data::QueryResults(ranked->num_results(), ranked->sorted_labels(),
        NULL));

    MetricScore orig_score = evaluate_result_list(results.get(),
                                                  ranked->sorted_scores());
    const unsigned int size = std::min(cutoff(), results->num_results());
    for (unsigned int i = 0; i < size; ++i) {
      double *p_jacobian = jacobian->vectat(i, i + 1);
      for (unsigned int j = i + 1; j < results->num_results(); ++j) {
        std::swap(ranked->sorted_scores()[i], ranked->sorted_scores()[j]);
        MetricScore new_score = evaluate_result_list(results.get(),
                                                     ranked->sorted_scores());
        *p_jacobian++ = new_score - orig_score;
        std::swap(ranked->sorted_scores()[i], ranked->sorted_scores()[j]);
      }
    }

    return jacobian;
  }

 private:

  /// The metric cutoff.
  unsigned int cutoff_;

  /// The output stream operator.
  friend std::ostream& operator<<(std::ostream& os, const Metric& m) {
    return m.put(os);
  }
  /// Prints the short name of the Metric, e.g., "NDCG@K"
  virtual std::ostream& put(std::ostream& os) const = 0;

};

}  // namespace ir
}  // namespace metric
}  // namespace quickrank

#endif
