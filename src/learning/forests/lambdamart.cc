/*
 * QuickRank - A C++ suite of Learning to Rank algorithms
 * Webpage: http://quickrank.isti.cnr.it/
 * Contact: quickrank@isti.cnr.it
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
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

LambdaMart::LambdaMart(const boost::property_tree::ptree &info_ptree,
                       const boost::property_tree::ptree &model_ptree)
    : Mart(info_ptree, model_ptree) {
  ntrees_ = 0;
  shrinkage_ = 0;
  nthresholds_ = 0;
  nleaves_ = 0;
  minleafsupport_ = 0;
  valid_iterations_ = 0;

  // read (training) info
  ntrees_ = info_ptree.get<unsigned int>("trees");
  nleaves_ = info_ptree.get<unsigned int>("leaves");
  minleafsupport_ = info_ptree.get<unsigned int>("leafsupport");
  nthresholds_ = info_ptree.get<unsigned int>("discretization");
  valid_iterations_ = info_ptree.get<unsigned int>("estop");
  shrinkage_ = info_ptree.get<double>("shrinkage");

  // read ensemble
  ensemble_model_.set_capacity(ntrees_);

  // loop over trees
  BOOST_FOREACH(const boost::property_tree::ptree::value_type& tree, model_ptree ){
  RTNode* root = NULL;
  float tree_weight = tree.second.get<double>("<xmlattr>.weight", shrinkage_);

  // find the root of the tree
  BOOST_FOREACH(const boost::property_tree::ptree::value_type& node, tree.second ) {
    if (node.first == "split") {
      root = io::RTNode_parse_xml(node.second);
      break;
    }
  }

  if (root == NULL) {
    std::cerr << "!!! Unable to parse tree from XML model." << std::endl;
    exit(EXIT_FAILURE);
  }

  ensemble_model_.push(root, tree_weight, -1);
}
}

std::ostream& LambdaMart::put(std::ostream& os) const {
  os << "# Ranker: " << name() << std::endl;
  os << "# max no. of trees = " << ntrees_ << std::endl;
  os << "# no. of tree leaves = " << nleaves_ << std::endl;
  os << "# shrinkage = " << shrinkage_ << std::endl;
  os << "# min leaf support = " << minleafsupport_ << std::endl;
  if (nthresholds_)
    os << "# no. of thresholds = " << nthresholds_ << std::endl;
  else
    os << "# no. of thresholds = unlimited" << std::endl;
  if (valid_iterations_)
    os << "# no. of no gain rounds before early stop = " << valid_iterations_
       << std::endl;
  return os;
}

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


std::ofstream& LambdaMart::save_model_to_file(std::ofstream& os) const {
  // write ranker description
  os << "\t<info>" << std::endl;
  os << "\t\t<type>" << name() << "</type>" << std::endl;
  os << "\t\t<trees>" << ntrees_ << "</trees>" << std::endl;
  os << "\t\t<leaves>" << nleaves_ << "</leaves>" << std::endl;
  os << "\t\t<shrinkage>" << shrinkage_ << "</shrinkage>" << std::endl;
  os << "\t\t<leafsupport>" << minleafsupport_ << "</leafsupport>" << std::endl;
  os << "\t\t<discretization>" << nthresholds_ << "</discretization>"
     << std::endl;
  os << "\t\t<estop>" << valid_iterations_ << "</estop>" << std::endl;
  os << "\t</info>" << std::endl;

  // save xml model
  ensemble_model_.save_model_to_file(os);

  return os;
}

}  // namespace forests
}  // namespace learning
}  // namespace quickrank
