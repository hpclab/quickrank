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
#include "learning/forests/randomforest.h"

#include <fstream>
#include <iomanip>

namespace quickrank {
namespace learning {
namespace forests {

const std::string RandomForest::NAME_ = "RANDOMFOREST";


void RandomForest::init(
    std::shared_ptr<quickrank::data::VerticalDataset> training_dataset) {

  Mart::init(training_dataset);

  const size_t nentries = training_dataset->num_instances();
  #pragma omp parallel for
  for (size_t i = 0; i < nentries; i++) {
    pseudoresponses_[i] = training_dataset->getLabel(i);
  }
}

void RandomForest::compute_pseudoresponses(
    std::shared_ptr<quickrank::data::VerticalDataset> training_dataset,
    quickrank::metric::ir::Metric *scorer) {

  // Do Nothing here, pseudoresponses_ does not change among iterations!
  return;
}

}  // namespace forests
}  // namespace learning
}  // namespace quickrank
