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
#include <optimization/optimization_factory.h>
#include "learning/ltr_algorithm_factory.h"

#include "learning/forests/mart.h"
#include "learning/forests/dart.h"
#include "learning/forests/lambdamart.h"
#include "learning/forests/obliviousmart.h"
#include "learning/forests/obliviouslambdamart.h"
#include "learning/forests/rankboost.h"
#include "learning/linear/coordinate_ascent.h"
#include "learning/linear/line_search.h"
#include "learning/custom/custom_ltr.h"

namespace quickrank {
namespace learning {

std::shared_ptr<quickrank::learning::LTR_Algorithm> ltr_algorithm_factory(
    ParamsMap &pmap) {

  std::shared_ptr<quickrank::learning::LTR_Algorithm> ltr_algo_model = nullptr;
  std::shared_ptr<quickrank::learning::LTR_Algorithm> ltr_algo = nullptr;

  // If it is set the source model (file) option, we need to load the LtR model
  if (pmap.isSet("model-in")) {
    std::string model_filename = pmap.get<std::string>("model-in");

    std::cout << "# Loading model from file " << model_filename << std::endl;

    ltr_algo_model =
        learning::LTR_Algorithm::load_model_from_file(model_filename);

    if (!ltr_algo_model)
      std::cerr << " !! Unable to load model from file." << std::endl;
  }

  if (!pmap.isSet("model-in") || pmap.isSet("restart-train")) {

    // We have to create a model from scratch
    std::string algo_name = pmap.get<std::string>("algo");
    std::transform(algo_name.begin(), algo_name.end(),
                   algo_name.begin(), ::toupper);

    if (algo_name == quickrank::learning::forests::LambdaMart::NAME_) {
      ltr_algo = std::shared_ptr<quickrank::learning::LTR_Algorithm>(
          new quickrank::learning::forests::LambdaMart(
              pmap.get<size_t>("num-trees"),
              pmap.get<double>("shrinkage"),
              pmap.get<size_t>("num-thresholds"),
              pmap.get<size_t>("num-leaves"),
              pmap.get<size_t>("min-leaf-support"),
              pmap.get<size_t>("end-after-rounds")
          ));
    } else if (algo_name == quickrank::learning::forests::Mart::NAME_) {
      ltr_algo = std::shared_ptr<quickrank::learning::LTR_Algorithm>(
          new quickrank::learning::forests::Mart(
              pmap.get<size_t>("num-trees"),
              pmap.get<double>("shrinkage"),
              pmap.get<size_t>("num-thresholds"),
              pmap.get<size_t>("num-leaves"),
              pmap.get<size_t>("min-leaf-support"),
              pmap.get<size_t>("end-after-rounds")
          ));
    } else if (algo_name == quickrank::learning::forests::Dart::NAME_) {
      ltr_algo = std::shared_ptr<quickrank::learning::LTR_Algorithm>(
          new quickrank::learning::forests::Dart(
              pmap.get<size_t>("num-trees"),
              pmap.get<double>("shrinkage"),
              pmap.get<size_t>("num-thresholds"),
              pmap.get<size_t>("num-leaves"),
              pmap.get<size_t>("min-leaf-support"),
              pmap.get<size_t>("end-after-rounds"),
              quickrank::learning::forests::Dart::get_sampling_type(
                  pmap.get<std::string>("sample-type")),
              quickrank::learning::forests::Dart::get_normalization_type(
                  pmap.get<std::string>("normalize-type")),
              quickrank::learning::forests::Dart::get_adaptive_type(
                  pmap.get<std::string>("adaptive-type")),
              pmap.get<double>("rate-drop"),
              pmap.get<double>("skip-drop"),
              pmap.isSet("keep-drop"),
              pmap.isSet("best-on-train"),
              pmap.get<double>("random-keep"),
              pmap.isSet("drop-on-best")
          ));
    } else if (algo_name
        == quickrank::learning::forests::ObliviousMart::NAME_) {
      ltr_algo = std::shared_ptr<quickrank::learning::LTR_Algorithm>(
          new quickrank::learning::forests::ObliviousMart(
              pmap.get<size_t>("num-trees"),
              pmap.get<double>("shrinkage"),
              pmap.get<size_t>("num-thresholds"),
              pmap.get<size_t>("tree-depth"),
              pmap.get<size_t>("min-leaf-support"),
              pmap.get<size_t>("end-after-rounds")
          ));
    } else if (algo_name
        == quickrank::learning::forests::ObliviousLambdaMart::NAME_) {
      ltr_algo = std::shared_ptr<quickrank::learning::LTR_Algorithm>(
          new quickrank::learning::forests::ObliviousLambdaMart(
              pmap.get<size_t>("num-trees"),
              pmap.get<double>("shrinkage"),
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
    } else if (algo_name
        == quickrank::learning::linear::CoordinateAscent::NAME_) {
      ltr_algo = std::shared_ptr<quickrank::learning::LTR_Algorithm>(
          new quickrank::learning::linear::CoordinateAscent(
              pmap.get <unsigned int> ("num-samples"),
              pmap.get<double>("window-size"),
              pmap.get<double>("reduction-factor"),
              pmap.get <unsigned int> ("max-iterations"),
              pmap.get <unsigned int> ("max-failed-valid")
          ));
    } else if (algo_name == quickrank::learning::linear::LineSearch::NAME_) {
      ltr_algo = std::shared_ptr<quickrank::learning::LTR_Algorithm>(
          new quickrank::learning::linear::LineSearch(
              pmap.get <unsigned int> ("num-samples"),
              pmap.get<double>("window-size"),
              pmap.get<double>("reduction-factor"),
              pmap.get <unsigned int > ("max-iterations"),
              pmap.get <unsigned int > ("max-failed-valid"),
              pmap.isSet("adaptive")
          ));
    } else if (algo_name == quickrank::learning::CustomLTR::NAME_) {
      ltr_algo = std::shared_ptr<quickrank::learning::LTR_Algorithm>(
          new quickrank::learning::CustomLTR());
    }
  }

  if (pmap.isSet("meta-algo")) {
    std::string meta_algo_name = pmap.get<std::string>("meta-algo");

    auto opt_algorithm =
        std::dynamic_pointer_cast<optimization::post_learning::pruning::Cleaver>(
            quickrank::optimization::optimization_factory(pmap));
    if (!opt_algorithm) {
      std::cerr << " !! Optimization Algorithm is required but it is not set "
          "properly" << std::endl;
      exit(EXIT_FAILURE);
    }

    if (meta_algo_name == quickrank::learning::meta::MetaCleaver::NAME_) {
      ltr_algo = std::shared_ptr<quickrank::learning::LTR_Algorithm>(
          new quickrank::learning::meta::MetaCleaver(
              ltr_algo,
              opt_algorithm,
              pmap.get<size_t>("final-num-trees"),
              pmap.get<size_t>("num-trees"),
              pmap.get<double>("pruning-rate"),
              pmap.isSet("opt-last-only"),
              pmap.get<size_t>("meta-end-after-rounds"),
              pmap.isSet("meta-verbose")
          )
      );
    }
  }

  if (ltr_algo_model) {
    if (ltr_algo && pmap.isSet("restart-train")) {
      bool res = ltr_algo->import_model_state(*ltr_algo_model);
      if (!res) {
        std::cerr << " !! Models not compatible for restart!" << std::endl;
        exit(EXIT_FAILURE);
      }
    } else
      ltr_algo = ltr_algo_model;
  }

  return ltr_algo;
}

}
}