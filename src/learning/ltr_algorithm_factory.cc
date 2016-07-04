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
#include "learning/ltr_algorithm_factory.h"

#include "learning/forests/mart.h"
#include "learning/forests/lambdamart.h"
#include "learning/forests/obliviousmart.h"
#include "learning/forests/obliviouslambdamart.h"
#include "learning/forests/rankboost.h"
#include "learning/linear/coordinate_ascent.h"
#include "learning/linear/line_search.h"
#include "learning/custom/custom_ltr.h"
#include "learning/linear/coordinate_ascent.h"

namespace quickrank {
namespace learning {

std::shared_ptr<quickrank::learning::LTR_Algorithm> ltr_algorithm_factory(
    ParamsMap& pmap, bool verbose) {

  std::shared_ptr<quickrank::learning::LTR_Algorithm> ltr_algo = nullptr;

  // If there is the model (file) option from the CLI but not the train option
  // it means we have to load an existing model from file in place of saving it
  if (pmap.isSet("model") && !pmap.isSet("train")) {
    std::string model_filename = pmap.get<std::string>("model");

    if (verbose)
      std::cout << "# Loading model from file " << model_filename << std::endl;

    ltr_algo = std::shared_ptr<quickrank::learning::LTR_Algorithm>(
            quickrank::learning::LTR_Algorithm::load_model_from_file(
                model_filename));

    if (verbose && !ltr_algo) {
      std::cerr << " !! Unable to load model from file." << std::endl;
    }

  } else {
    // We have to create a model from scratch
    std::string algo_name = pmap.get<std::string>("algo");
    std::transform(algo_name.begin(), algo_name.end(),
                   algo_name.begin(), ::toupper);

    if (algo_name == quickrank::learning::forests::LambdaMart::NAME_) {
      ltr_algo = std::shared_ptr<quickrank::learning::LTR_Algorithm>(
          new quickrank::learning::forests::LambdaMart(
              pmap.get<size_t>("num-trees"),
              pmap.get<size_t>("shrinkage"),
              pmap.get<size_t>("num-thresholds"),
              pmap.get<size_t>("num-leaves"),
              pmap.get<size_t>("min-leaf-support"),
              pmap.get<size_t>("end-after-rounds")
          ));
    } else if (algo_name == quickrank::learning::forests::LambdaMart::NAME_) {
      ltr_algo = std::shared_ptr<quickrank::learning::LTR_Algorithm>(
          new quickrank::learning::forests::Mart(
              pmap.get<size_t>("num-trees"),
              pmap.get<size_t>("shrinkage"),
              pmap.get<size_t>("num-thresholds"),
              pmap.get<size_t>("num-leaves"),
              pmap.get<size_t>("min-leaf-support"),
              pmap.get<size_t>("end-after-rounds")
          ));
    } else if (algo_name == quickrank::learning::forests::ObliviousMart::NAME_) {
      ltr_algo = std::shared_ptr<quickrank::learning::LTR_Algorithm>(
          new quickrank::learning::forests::ObliviousMart(
              pmap.get<size_t>("num-trees"),
              pmap.get<size_t>("shrinkage"),
              pmap.get<size_t>("num-thresholds"),
              pmap.get<size_t>("tree-depth"),
              pmap.get<size_t>("min-leaf-support"),
              pmap.get<size_t>("end-after-rounds")
          ));
    } else if (algo_name == quickrank::learning::forests::ObliviousLambdaMart::NAME_) {
      ltr_algo = std::shared_ptr<quickrank::learning::LTR_Algorithm>(
          new quickrank::learning::forests::ObliviousLambdaMart(
              pmap.get<size_t>("num-trees"),
              pmap.get<size_t>("shrinkage"),
              pmap.get<size_t>("num-thresholds"),
              pmap.get<size_t>("tree-depth"),
              pmap.get<size_t>("min-leaf-support"),
              pmap.get<size_t>("end-after-rounds")
          ));
    } else if (algo_name == quickrank::learning::forests::Rankboost::NAME_) {
      ltr_algo = std::shared_ptr<quickrank::learning::LTR_Algorithm>(
          new quickrank::learning::forests::Rankboost(
              pmap.get<size_t>("num-trees")
          ));
    } else if (algo_name == quickrank::learning::linear::CoordinateAscent::NAME_) {
      ltr_algo = std::shared_ptr<quickrank::learning::LTR_Algorithm>(
          new quickrank::learning::linear::CoordinateAscent(
              pmap.get<size_t>("num-samples"),
              pmap.get<float>("window-size"),
              pmap.get<float>("reduction-factor"),
              pmap.get<size_t>("max-iterations"),
              pmap.get<size_t>("max-failed-valid")
          ));
    } else if (algo_name == quickrank::learning::linear::LineSearch::NAME_) {
      ltr_algo = std::shared_ptr<quickrank::learning::LTR_Algorithm>(
          new quickrank::learning::linear::LineSearch(
              pmap.get<size_t>("num-samples"),
              pmap.get<float>("window-size"),
              pmap.get<float>("reduction-factor"),
              pmap.get<size_t>("max-iterations"),
              pmap.get<size_t>("max-failed-valid"),
              pmap.isSet("adaptive")
          ));
    } else if (algo_name == quickrank::learning::CustomLTR::NAME_) {
      ltr_algo = std::shared_ptr<quickrank::learning::LTR_Algorithm>(
          new quickrank::learning::CustomLTR());
    }
  }

  return ltr_algo;
}

}
}