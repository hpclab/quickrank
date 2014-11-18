#include "learning/forests/mart.h"

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

std::ostream& Mart::put(std::ostream& os) const {
  os << "# Ranker: MART" << std::endl
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


void Mart::compute_pseudoresponses( std::shared_ptr<quickrank::data::Dataset> training_dataset,
                                          quickrank::metric::ir::Metric* scorer) {
   const unsigned int nentries = training_dataset->num_instances();
   for(unsigned int i=0; i<nentries; i++)
     pseudoresponses[i] = training_dataset->getLabel(i) - trainingmodelscores[i];
}

float Mart::update_tree_prediction( RegressionTree* tree) {
  return tree->update_output(pseudoresponses);
}

}  // namespace forests
}  // namespace learning
}  // namespace quickrank
