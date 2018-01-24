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
#include "learning/forests/lambdamart.h"

#include <fstream>
#include <iomanip>

namespace quickrank {
namespace learning {
namespace forests {

const std::string LambdaMart::NAME_ = "LAMBDAMART";


void
LambdaMart::init(std::shared_ptr<quickrank::data::VerticalDataset> training_dataset) {
  Mart::init(training_dataset);
  const size_t nentries = training_dataset->num_instances();
  instance_weights_ = new double[nentries]();  //0.0f initialized
}

void LambdaMart::clear(size_t num_features) {
  Mart::clear(num_features);
  if (instance_weights_)
    delete[] instance_weights_;
}

std::unique_ptr<RegressionTree> LambdaMart::fit_regressor_on_gradient(
    std::shared_ptr<data::VerticalDataset> training_dataset,
    size_t *sampleids) {
  //Fit a regression tree
  /// \todo TODO: memory management of regression tree is wrong!!!
  RegressionTree *tree = new RegressionTree(nleaves_, training_dataset.get(),
                                            pseudoresponses_, minleafsupport_,
                                            collapse_leaves_factor_);
  tree->fit(hist_, sampleids, max_features_);
  //update the outputs of the tree (with gamma computed using the Newton-Raphson pruning_method)
  //float maxlabel =
  tree->update_output(pseudoresponses_, instance_weights_);

  return std::unique_ptr<RegressionTree>(tree);
}

void LambdaMart::compute_pseudoresponses(
    std::shared_ptr<quickrank::data::VerticalDataset> training_dataset,
    quickrank::metric::ir::Metric *scorer,
    bool *sample_presence) {

  const size_t cutoff = scorer->cutoff();

  const size_t nrankedlists = training_dataset->num_queries();
  #pragma omp parallel for
  for (size_t i = 0; i < nrankedlists; ++i) {
    std::shared_ptr<data::QueryResults> qr =
        training_dataset->getQueryResults(i);

    const size_t offset = training_dataset->offset(i);
    // Reset pseudoresponses and instance_weights before updating...
    for (size_t j = offset; j < offset + qr->num_results(); ++j)
      pseudoresponses_[j] = instance_weights_[j] = 0.0;


    std::shared_ptr<data::RankedResults> ranked;
    size_t *map_from_cleaned = new size_t[qr->num_results()];
    Label *labels_cleaned;
    Score *training_scores_cleaned;
    if (sample_presence) {
      // Clean the query results with missing samples
      labels_cleaned = new Label[qr->num_results()];
      training_scores_cleaned = new Score[qr->num_results()];
      size_t count = 0;
      for (size_t d = 0; d < qr->num_results(); ++d) {
        if (sample_presence[offset + d]) {
          map_from_cleaned[count] = d;
          labels_cleaned[count] = qr->labels()[d];
          training_scores_cleaned[count] = scores_on_training_[d];
          ++count;
        }
      }

      auto qr_cleaned = std::shared_ptr<data::QueryResults>(
          new data::QueryResults(count, labels_cleaned, NULL));
      ranked = std::shared_ptr<data::RankedResults>(
          new data::RankedResults(qr_cleaned, training_scores_cleaned));
    } else {
      for (size_t d = 0; d < qr->num_results(); ++d)
        map_from_cleaned[d] = d;
      ranked = std::shared_ptr<data::RankedResults>(
          new data::RankedResults(qr, scores_on_training_ + offset));
    }

    std::unique_ptr<Jacobian> jacobian = scorer->jacobian(ranked);

    // \todo TODO: rank by label once and for all ?
    // \todo TODO: avoid n^2 loop ?
    for (size_t j = 0; j < ranked->num_results(); j++) {
      Label jthlabel = ranked->sorted_labels()[j];

      size_t j_abs = offset + map_from_cleaned[ranked->pos_of_rank(j)];

      for (size_t k = 0; k < ranked->num_results(); k++) {

        size_t k_abs = offset + map_from_cleaned[ranked->pos_of_rank(k)];

        if (k != j) {
          // skip if we are beyond the top-K results
          if (j >= cutoff && k >= cutoff)
            break;

          Label kthlabel = ranked->sorted_labels()[k];
          if (jthlabel > kthlabel) {
            double deltandcg = fabs(jacobian->at(j, k));

            double rho = 1.0
                / (1.0 + exp(scores_on_training_[j_abs]
                                 - scores_on_training_[k_abs]) );
            double lambda = rho * deltandcg;
            double delta = rho * (1.0 - rho) * deltandcg;
            pseudoresponses_[j_abs] += lambda;
            pseudoresponses_[k_abs] -= lambda;
            instance_weights_[j_abs] += delta;
            instance_weights_[k_abs] += delta;
          }
        }
      }
    }

    if (sample_presence) {
      delete[] labels_cleaned;
      delete[] training_scores_cleaned;
      delete[] map_from_cleaned;
    }
  }
}

}  // namespace forests
}  // namespace learning
}  // namespace quickrank
