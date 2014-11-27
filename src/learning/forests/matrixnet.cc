#include "learning/forests/matrixnet.h"

#include <iostream>
#include <fstream>
#include <iomanip>
#include <cfloat>
#include <cmath>
#include <boost/foreach.hpp>

#include "utils/qsort.h"
#include "io/xml.h"

namespace quickrank {
namespace learning {
namespace forests {


MatrixNet::MatrixNet(const boost::property_tree::ptree &info_ptree,
                     const boost::property_tree::ptree &model_ptree)
    : LambdaMart(info_ptree, model_ptree) {
  ntrees_ = 0;
  shrinkage_ = 0;
  nthresholds_ = 0;
  nleaves_ = 0;
  minleafsupport_ = 0;
  valid_iterations_ = 0;
  treedepth_ = 0;

  // read (training) info
  ntrees_ = info_ptree.get<unsigned int>("trees");
  minleafsupport_ = info_ptree.get<unsigned int>("leafsupport");
  nthresholds_ = info_ptree.get<unsigned int>("discretization");
  valid_iterations_ = info_ptree.get<unsigned int>("estop");
  shrinkage_ = info_ptree.get<double>("shrinkage");
  treedepth_ = info_ptree.get<double>("depth");

  // read ensemble
  ensemble_model_.set_capacity(ntrees_);

  // loop over trees
  BOOST_FOREACH(const boost::property_tree::ptree::value_type& tree, model_ptree) {
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

std::ostream& MatrixNet::put(std::ostream& os) const {
  os << "# Ranker: MatrixNet" << std::endl << "# max no. of trees = " << ntrees_
     << std::endl << "# max tree depth = " << treedepth_ << std::endl
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

std::unique_ptr<RegressionTree> MatrixNet::fit_regressor_on_gradient(
    std::shared_ptr<data::Dataset> training_dataset) {
  ObliviousRT* tree = new ObliviousRT(nleaves_, training_dataset.get(),
                                      pseudoresponses_, minleafsupport_,
                                      treedepth_);
  tree->fit(hist_);
  //update the outputs of the tree (with gamma computed using the Newton-Raphson method)
  tree->update_output(pseudoresponses_, instance_weights_);
  return std::unique_ptr<RegressionTree>(tree);
}

std::ofstream& MatrixNet::save_model_to_file(std::ofstream& os) const {
  // write ranker description
  os << "\t<info>" << std::endl << "\t\t<type>" << name() << "</type>"
     << std::endl << "\t\t<trees>" << ntrees_ << "</trees>" << std::endl
     << "\t\t<leaves>" << nleaves_ << "</leaves>" << std::endl << "\t\t<depth>"
     << treedepth_ << "</depth>" << std::endl << "\t\t<shrinkage>" << shrinkage_
     << "</shrinkage>" << std::endl << "\t\t<leafsupport>" << minleafsupport_
     << "</leafsupport>" << std::endl << "\t\t<discretization>" << nthresholds_
     << "</discretization>" << std::endl << "\t\t<estop>" << valid_iterations_
     << "</estop>" << std::endl << "\t</info>" << std::endl;

  // save xml model
  ensemble_model_.save_model_to_file(os);

  return os;
}

}  // namespace forests
}  // namespace learning
}  // namespace quickrank
