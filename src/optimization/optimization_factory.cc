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
 * Contributors:
 *  - Salvatore Trani(salvatore.trani@isti.cnr.it)
 */

#include "optimization/optimization_factory.h"
#include "optimization/post_learning/pruning/ensemble_pruning.h"
#include "optimization/post_learning/pruning/ensemble_pruning_factory.h"

namespace quickrank {
namespace optimization {
std::shared_ptr<learning::linear::LineSearch> linesearch_opt_factory(
    ParamsMap& pmap) {

  std::shared_ptr<learning::linear::LineSearch> lineSearch = nullptr;

  // Check if line search is to do...
  if (pmap.isSet("with-line-search")) {

    if (pmap.isSet("line-search-model")) {
      // We have to load the model from file
      std::string xml_linesearch_filename =
          pmap.get<std::string>("line-search-model");

      lineSearch =
          std::dynamic_pointer_cast<quickrank::learning::linear::LineSearch>(
              quickrank::learning::LTR_Algorithm::load_model_from_file(
              xml_linesearch_filename));

    } else {
      // We have to create an empty model
      lineSearch = std::shared_ptr<learning::linear::LineSearch>(
          new quickrank::learning::linear::LineSearch(
              pmap.get<size_t>("num-samples"),
              pmap.get<float>("window-size"),
              pmap.get<float>("reduction-factor"),
              pmap.get<size_t>("max-iterations"),
              pmap.get<size_t>("max-failed-valid"),
              pmap.isSet("adaptive")
      ));
    }
  }

  return lineSearch;
}

std::shared_ptr<quickrank::optimization::Optimization> optimization_factory(
    ParamsMap& pmap) {

  std::shared_ptr<quickrank::optimization::Optimization> optimizer = nullptr;
  std::shared_ptr<learning::linear::LineSearch> lineSearch =
      linesearch_opt_factory(pmap);

  if (pmap.isSet("opt_algo")) {

    std::string opt_algo = pmap.get<std::string>("opt_algo");

    if (opt_algo == quickrank::optimization::post_learning::pruning
                    ::EnsemblePruning::NAME_) {

      optimizer = quickrank::optimization::post_learning::pruning::create_pruner(
          pmap.get<std::string>("opt-method"),
          pmap.get<double>("pruning-rate"),
          lineSearch
      );
    }
  }

  return optimizer;
}

}
}
