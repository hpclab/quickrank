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

#include <iostream>
#include <fstream>
#include <iomanip>
#include <cfloat>
#include <cmath>
#include <boost/foreach.hpp>

#include "data/rankedresults.h"
#include "io/xml.h"

namespace quickrank {
namespace learning {
namespace forests {

const std::string LambdaMart::NAME_ = "LAMBDAMART";

void LambdaMart::init(
    std::shared_ptr<quickrank::data::Dataset> training_dataset,
    std::shared_ptr<quickrank::data::Dataset> validation_dataset) {
  Mart::init(training_dataset, validation_dataset);
  const unsigned int nentries = training_dataset->num_instances();
  instance_weights_ = new double[nentries]();  //0.0f initialized
}

void LambdaMart::clear(std::shared_ptr<data::Dataset> training_dataset) {
  Mart::clear(training_dataset);
  if (instance_weights_)
    delete[] instance_weights_;
}

std::unique_ptr<RegressionTree> LambdaMart::fit_regressor_on_gradient(
    std::shared_ptr<data::Dataset> training_dataset) {
  //Fit a regression tree
  /// \todo TODO: memory management of regression tree is wrong!!!
  RegressionTree* tree = new RegressionTree(nleaves_, training_dataset.get(),
                                            pseudoresponses_, minleafsupport_);
  tree->fit(hist_);
  //update the outputs of the tree (with gamma computed using the Newton-Raphson method)
  //float maxlabel =
  tree->update_output(pseudoresponses_, instance_weights_);

  return std::unique_ptr<RegressionTree>(tree);
}

void LambdaMart::compute_pseudoresponses(
    std::shared_ptr<quickrank::data::Dataset> training_dataset,
    quickrank::metric::ir::Metric* scorer) {
  const unsigned int cutoff = scorer->cutoff();

  const unsigned int nrankedlists = training_dataset->num_queries();
#pragma omp parallel for
  for (unsigned int i = 0; i < nrankedlists; ++i) {
    std::shared_ptr<data::QueryResults> qr = training_dataset->getQueryResults(
        i);

    const unsigned int offset = training_dataset->offset(i);
    double *lambdas = pseudoresponses_ + offset;
    double *weights = instance_weights_ + offset;
    for (unsigned int j = 0; j < qr->num_results(); ++j)
      lambdas[j] = weights[j] = 0.0;

    auto ranked = std::shared_ptr<data::RankedResults>(
        new data::RankedResults(qr, scores_on_training_ + offset));

    std::unique_ptr<Jacobian> jacobian = scorer->jacobian(ranked);

    // \todo TODO: rank by label once and for all ?
    // \todo TODO: avoid n^2 loop ?
    for (unsigned int j = 0; j < ranked->num_results(); j++) {
      Label jthlabel = ranked->sorted_labels()[j];
      for (unsigned int k = 0; k < ranked->num_results(); k++)
        if (k != j) {
          // skip if we are beyond the top-K results
          if (j >= cutoff && k >= cutoff)
            break;

          Label kthlabel = ranked->sorted_labels()[k];
          if (jthlabel > kthlabel) {
            double deltandcg = fabs(jacobian->at(j, k));

            double rho = 1.0
                / (1.0
                    + exp(
                        scores_on_training_[offset + ranked->pos_of_rank(j)]
                            - scores_on_training_[offset
                                + ranked->pos_of_rank(k)]));
            double lambda = rho * deltandcg;
            double delta = rho * (1.0 - rho) * deltandcg;
            lambdas[ranked->pos_of_rank(j)] += lambda;
            lambdas[ranked->pos_of_rank(k)] -= lambda;
            weights[ranked->pos_of_rank(j)] += delta;
            weights[ranked->pos_of_rank(k)] += delta;
          }
        }
    }
  }
}


}  // namespace forests
}  // namespace learning
}  // namespace quickrank
