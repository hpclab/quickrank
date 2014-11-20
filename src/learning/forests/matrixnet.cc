#include "learning/forests/matrixnet.h"

#include <iostream>
#include <iomanip>
#include <cfloat>
#include <cmath>

#include "utils/qsort.h"

namespace quickrank {
namespace learning {
namespace forests {

std::ostream& MatrixNet::put(std::ostream& os) const {
  os  << "# Ranker: MatrixNet" << std::endl
      << "# max no. of trees = " << ntrees_ << std::endl
      << "# max tree depth = " << treedepth_ << std::endl
      << "# shrinkage = " << shrinkage_ << std::endl
      << "# min leaf support = " << minleafsupport_ << std::endl;
  if (nthresholds_)
    os << "# no. of thresholds = " << nthresholds_ << std::endl;
  else
    os << "# no. of thresholds = unlimited" << std::endl;
  if (valid_iterations_)
    os << "# no. of no gain rounds before early stop = " << valid_iterations_ << std::endl;
  return os;
}

std::unique_ptr<RegressionTree> MatrixNet::fit_regressor_on_gradient (
    std::shared_ptr<data::Dataset> training_dataset ) {
  ObliviousRT* tree = new ObliviousRT(nleaves_, training_dataset.get(),
                                      pseudoresponses_, minleafsupport_,
                                      treedepth_);
  tree->fit(hist_);
  //update the outputs of the tree (with gamma computed using the Newton-Raphson method)
  tree->update_output(pseudoresponses_, instance_weights_);
  return std::unique_ptr<RegressionTree>(tree);
}



}  // namespace forests
}  // namespace learning
}  // namespace quickrank
