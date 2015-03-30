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
#include "learning/forests/obliviouslambdamart.h"

#include <iostream>
#include <fstream>
#include <iomanip>
#include <cfloat>
#include <cmath>
#include <boost/foreach.hpp>

#include "io/xml.h"

namespace quickrank {
namespace learning {
namespace forests {

const std::string ObliviousLambdaMart::NAME_ = "OBVLAMBDAMART";


ObliviousLambdaMart::ObliviousLambdaMart(const boost::property_tree::ptree &info_ptree,
                     const boost::property_tree::ptree &model_ptree)
    : LambdaMart(info_ptree, model_ptree) {
  treedepth_ = info_ptree.get<double>("depth");
}

std::ostream& ObliviousLambdaMart::put(std::ostream& os) const {
  os << "# Ranker: "<< name() << std::endl << "# max no. of trees = " << ntrees_
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

std::unique_ptr<RegressionTree> ObliviousLambdaMart::fit_regressor_on_gradient(
    std::shared_ptr<data::Dataset> training_dataset) {
  ObliviousRT* tree = new ObliviousRT(nleaves_, training_dataset.get(),
                                      pseudoresponses_, minleafsupport_,
                                      treedepth_);
  tree->fit(hist_);
  //update the outputs of the tree (with gamma computed using the Newton-Raphson method)
  tree->update_output(pseudoresponses_, instance_weights_);
  return std::unique_ptr<RegressionTree>(tree);
}

std::ofstream& ObliviousLambdaMart::save_model_to_file(std::ofstream& os) const {
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
