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
#include "learning/forests/obliviousmart.h"

#include <fstream>
#include <iomanip>

namespace quickrank {
namespace learning {
namespace forests {

const std::string ObliviousMart::NAME_ = "OBVMART";

ObliviousMart::ObliviousMart(const pugi::xml_document &model) : Mart(model) {

  treedepth_ = model.child("ranker").child("info").child("depth").text()
      .as_int();
}

std::ostream &ObliviousMart::put(std::ostream &os) const {
  os << "# Ranker: " << name() << std::endl << "# max no. of trees = "
     << ntrees_ << std::endl << "# max tree depth = " << treedepth_
     << std::endl
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

std::unique_ptr<RegressionTree> ObliviousMart::fit_regressor_on_gradient(
    std::shared_ptr<data::VerticalDataset> training_dataset) {
  ObliviousRT *tree = new ObliviousRT(nleaves_, training_dataset.get(),
                                      pseudoresponses_, minleafsupport_,
                                      treedepth_);
  tree->fit(hist_);
  //update the outputs of the tree (with gamma computed using the Newton-Raphson pruning_method)
  tree->update_output(pseudoresponses_);
  return std::unique_ptr<RegressionTree>(tree);
}

pugi::xml_document *ObliviousMart::get_xml_model() const {

  pugi::xml_document *doc = new pugi::xml_document();
  pugi::xml_node root = doc->append_child("ranker");

  pugi::xml_node info = root.append_child("info");

  info.append_child("type").text() = name().c_str();
  info.append_child("trees").text() = ntrees_;
  info.append_child("leaves").text() = nleaves_;
  info.append_child("depth").text() = treedepth_;
  info.append_child("shrinkage").text() = shrinkage_;
  info.append_child("leafsupport").text() = minleafsupport_;
  info.append_child("discretization").text() = nthresholds_;
  info.append_child("estop").text() = nthresholds_;

  ensemble_model_.append_xml_model(root);

  return doc;
}

bool ObliviousMart::import_model_state(LTR_Algorithm &other) {

  // Check the object is derived from Mart
  try
  {
    ObliviousMart& otherCast = dynamic_cast<ObliviousMart&>(other);

    if (treedepth_ != otherCast.treedepth_)
      return false;

    // Call the super method in Mart
    return Mart::import_model_state(other);
  }
  catch(std::bad_cast)
  {
    return false;
  }
}

}  // namespace forests
}  // namespace learning
}  // namespace quickrank
