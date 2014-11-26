#include "learning/forests/lambdamart.h"

#include <iostream>
#include <fstream>
#include <iomanip>
#include <cfloat>
#include <cmath>
#include <boost/foreach.hpp>

#include "utils/radix.h"
#include "utils/qsort.h"
#include "utils/mergesorter.h"
#include "data/rankedresults.h"

namespace quickrank {
namespace learning {
namespace forests {

RTNode* RTNode_parse_xml(const boost::property_tree::ptree &split_xml) {
  RTNode* model_node = NULL;
  RTNode* left_child = NULL;
  RTNode* right_child = NULL;

  bool is_leaf = false;

  unsigned int feature_id = 0;
  float threshold = 0.0f;
  double prediction = 0.0;

  BOOST_FOREACH(const boost::property_tree::ptree::value_type& split_child, split_xml ) {
    if (split_child.first=="output") {
      prediction = split_child.second.get_value<double>();
      is_leaf = true;
      break;
    } else if (split_child.first=="feature") {
      feature_id = split_child.second.get_value<unsigned int>();
    } else if (split_child.first=="threshold") {
      threshold = split_child.second.get_value<float>();
    } else if (split_child.first=="split") {
      std::string pos = split_child.second.get<std::string>("<xmlattr>.pos");
      if (pos=="left")
      left_child = RTNode_parse_xml(split_child.second);
      else
      right_child = RTNode_parse_xml(split_child.second);
    }
  }

  if (is_leaf)
    model_node = new RTNode(prediction);
  else
    /// \todo TODO: this should be changed with item mapping
    model_node = new RTNode(threshold, feature_id - 1, feature_id, left_child,
                            right_child);

  return model_node;
}

LambdaMart::LambdaMart(const boost::property_tree::ptree &info_ptree,
           const boost::property_tree::ptree &model_ptree) {
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
BOOST_FOREACH(const boost::property_tree::ptree::value_type& tree, model_ptree) {
  RTNode* root = NULL;
  float tree_weight = tree.second.get<double>("<xmlattr>.weight", shrinkage_);

  // find the root of the tree
  BOOST_FOREACH(const boost::property_tree::ptree::value_type& node, tree.second ) {
    if (node.first=="split") {
      root = RTNode_parse_xml(node.second);
      break;
    }
  }

  if (root==NULL) {
    std::cerr << "!!! Unable to parse tree from XML model." << std::endl;
    exit(EXIT_FAILURE);
  }

  ensemble_model_.push(root, tree_weight, -1);
}
}

std::ostream& LambdaMart::put(std::ostream& os) const {
os << "# Ranker: Lambda-MART" << std::endl << "# max no. of trees = " << ntrees_
   << std::endl << "# no. of tree leaves = " << nleaves_ << std::endl
   << "# shrinkage = " << shrinkage_ << std::endl << "# min leaf support = "
   << minleafsupport_ << std::endl;
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
  std::shared_ptr<data::QueryResults> qr = training_dataset->getQueryResults(i);

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

          double rho =
              1.0
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

std::ofstream& LambdaMart::save_model_to_file(std::ofstream& os) const {
// write ranker description
os << "\t<info>" << std::endl << "\t\t<type>" << name() << "</type>"
   << std::endl << "\t\t<trees>" << ntrees_ << "</trees>" << std::endl
   << "\t\t<leaves>" << nleaves_ << "</leaves>" << std::endl
   << "\t\t<shrinkage>" << shrinkage_ << "</shrinkage>" << std::endl
   << "\t\t<leafsupport>" << minleafsupport_ << "</leafsupport>" << std::endl
   << "\t\t<discretization>" << nthresholds_ << "</discretization>" << std::endl
   << "\t\t<estop>" << valid_iterations_ << "</estop>" << std::endl
   << "\t</info>" << std::endl;

// save xml model
ensemble_model_.save_model_to_file(os);

return os;
}

}  // namespace forests
}  // namespace learning
}  // namespace quickrank
