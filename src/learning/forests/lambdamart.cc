#include "learning/forests/lambdamart.h"

#include <iostream>
#include <fstream>
#include <iomanip>
#include <cfloat>
#include <cmath>

#include "utils/radix.h"
#include "utils/qsort.h"
#include "utils/mergesorter.h"
#include "data/rankedresults.h"

namespace quickrank {
namespace learning {
namespace forests {

std::ostream& LambdaMart::put(std::ostream& os) const {
  os << "# Ranker: Lambda-MART" << std::endl
      << "# max no. of trees = " << ntrees << std::endl
      << "# no. of tree leaves = " << ntreeleaves << std::endl
      << "# shrinkage = " << shrinkage << std::endl
      << "# min leaf support = " << minleafsupport << std::endl;
  if (nthresholds)
    os << "# no. of thresholds = " << nthresholds << std::endl;
  else
    os << "# no. of thresholds = unlimited" << std::endl;
  if (esr)
    os << "# no. of no gain rounds before early stop = " << esr << std::endl;
  return os;
}


void LambdaMart::init(std::shared_ptr<quickrank::data::Dataset> training_dataset,
                      std::shared_ptr<quickrank::data::Dataset> validation_dataset) {
  Mart::init(training_dataset, validation_dataset);
  const unsigned int nentries = training_dataset->num_instances();
  cachedweights = new double[nentries]();  //0.0f initialized
}


std::unique_ptr<RegressionTree> LambdaMart::fit_regressor_on_gradient (
    std::shared_ptr<data::Dataset> training_dataset ) {
  //Fit a regression tree
  /// \todo TODO: memory management of regression tree is wrong!!!
  RegressionTree* tree = new RegressionTree ( ntreeleaves, training_dataset.get(),
                                              pseudoresponses, minleafsupport);
  tree->fit(hist);
  //update the outputs of the tree (with gamma computed using the Newton-Raphson method)
  //float maxlabel =
  tree->update_output(pseudoresponses, cachedweights);
  return std::unique_ptr<RegressionTree>(tree);
}

void LambdaMart::compute_pseudoresponses( std::shared_ptr<quickrank::data::Dataset> training_dataset,
                                          quickrank::metric::ir::Metric* scorer) {
  const unsigned int cutoff = scorer->cutoff();

  const unsigned int nrankedlists = training_dataset->num_queries();
  #pragma omp parallel for
  for (unsigned int i = 0; i < nrankedlists; ++i) {
    std::shared_ptr<data::QueryResults> qr = training_dataset->getQueryResults(i);

    const unsigned int offset = training_dataset->offset(i);
    double *lambdas = pseudoresponses + offset;
    double *weights = cachedweights + offset;
    for (unsigned int j = 0; j < qr->num_results(); ++j)
      lambdas[j] = weights[j] = 0.0;

    auto ranked = std::shared_ptr<data::RankedResults>( new data::RankedResults(qr, trainingmodelscores + offset) );

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

            double rho = 1.0 / (1.0 + exp(
                        trainingmodelscores[offset + ranked->pos_of_rank(j)] -
                        trainingmodelscores[offset + ranked->pos_of_rank(k)] ));
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


// Changes by Cla:
// - added processing of ranked list in ranked order
// - added cut-off in measure changes matrix
/*
void LambdaMart::compute_pseudoresponses(std::shared_ptr<quickrank::data::Dataset> training_dataset, quickrank::metric::ir::Metric* scorer) {
  const unsigned int cutoff = scorer->cutoff();

  const unsigned int nrankedlists = training_dataset->num_queries();
  //const unsigned int *rloffsets = training_set->get_rloffsets();
  #pragma omp parallel for
  for (unsigned int i = 0; i < nrankedlists; ++i) {
    std::shared_ptr<data::QueryResults> qr = training_dataset->getQueryResults(i);

    const unsigned int offset = training_dataset->offset(i);
    double *lambdas = pseudoresponses + offset;
    double *weights = cachedweights + offset;
    for (unsigned int j = 0; j < qr->num_results(); ++j)
      lambdas[j] = weights[j] = 0.0;

    // CLA: line below uses the old sort and not mergesort as in ranklib
    // unsigned int *idx = idxdouble_qsort(trainingmodelscores+offset, ql.size);
    unsigned int *idx = idxdouble_mergesort<Score>(
        trainingmodelscores + offset, qr->num_results());

    Label* sortedlabels = new Label[qr->num_results()];
    for (unsigned int i = 0; i < qr->num_results(); ++i)
      sortedlabels[i] = qr->labels()[idx[i]];

    std::shared_ptr<data::QueryResults> ranked_list = std::shared_ptr<data::QueryResults>(
        new data::QueryResults(qr->num_results(), sortedlabels, NULL) );

    std::unique_ptr<Jacobian> changes = scorer->get_jacobian(ranked_list);

    // \todo TODO: rank by label one and for all ?
    // \todo TODO: look at the top score or at the top labelled ?
    for (unsigned int j = 0; j < ranked_list->num_results(); ++j) {
      float jthlabel = ranked_list->labels()[j];
      for (unsigned int k = 0; k < ranked_list->num_results(); ++k)
        if (k != j) {
          // skip if we are beyond the top-K results
          if (j >= cutoff && k >= cutoff)
            break;

          float kthlabel = ranked_list->labels()[k];
          if (jthlabel > kthlabel) {
            int i_max = j >= k ? j : k;
            int i_min = j >= k ? k : j;
            double deltandcg = fabs(changes->at(i_min, i_max));

            double rho = 1.0 / (1.0 + exp(
                        trainingmodelscores[offset + idx[j]] -
                        trainingmodelscores[offset + idx[k]] ));
            double lambda = rho * deltandcg;
            double delta = rho * (1.0 - rho) * deltandcg;
            lambdas[idx[j]] += lambda;
            lambdas[idx[k]] -= lambda;
            weights[idx[j]] += delta;
            weights[idx[k]] += delta;
          }
        }
    }

    delete[] idx;
    delete[] sortedlabels;

  }
}
*/


}  // namespace forests
}  // namespace learning
}  // namespace quickrank
