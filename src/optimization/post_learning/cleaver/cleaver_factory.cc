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
#include "optimization/post_learning/cleaver/cleaver_factory.h"

#include "optimization/post_learning/cleaver/random_pruning.h"
#include "optimization/post_learning/cleaver/random_adv_pruning.h"
#include "optimization/post_learning/cleaver/last_pruning.h"
#include "optimization/post_learning/cleaver/skip_pruning.h"
#include "optimization/post_learning/cleaver/score_loss_pruning.h"
#include "optimization/post_learning/cleaver/low_weights_pruning.h"
#include "optimization/post_learning/cleaver/quality_loss_pruning.h"
#include "optimization/post_learning/cleaver/quality_loss_adv_pruning.h"

namespace quickrank {
namespace optimization {
namespace post_learning {
namespace pruning {

std::shared_ptr<quickrank::optimization::Optimization> create_pruner(
    const pugi::xml_document& model) {

  pugi::xml_node model_info = model.child("optimizer").child("info");

  std::string pruningMethodName =
      model_info.child("opt-method").text().as_string();

  if (pruningMethodName.empty()) {
    std::cerr << "Unable to find the name of the optimization method inside "
        "the xml." << std::endl;
    exit(EXIT_FAILURE);
  }


  Cleaver::PruningMethod pruningMethod =
      Cleaver::get_pruning_method(pruningMethodName);

  Optimization *optimizer = nullptr;

  switch (pruningMethod) {
    case Cleaver::PruningMethod::RANDOM: {
      optimizer = new RandomPruning(model);
      break;
    }
    case Cleaver::PruningMethod::RANDOM_ADV: {
      optimizer = new RandomAdvPruning(model);
      break;
    }
    case Cleaver::PruningMethod::LOW_WEIGHTS: {
      optimizer = new LowWeightsPruning(model);
      break;
    }
    case Cleaver::PruningMethod::LAST: {
      optimizer = new LastPruning(model);
      break;
    }
    case Cleaver::PruningMethod::QUALITY_LOSS: {
      optimizer = new QualityLossPruning(model);
      break;
    }
    case Cleaver::PruningMethod::QUALITY_LOSS_ADV: {
      optimizer = new QualityLossAdvPruning(model);
      break;
    }
    case Cleaver::PruningMethod::SKIP: {
      optimizer = new SkipPruning(model);
      break;
    }
    case Cleaver::PruningMethod::SCORE_LOSS: {
      optimizer = new ScoreLossPruning(model);
      break;
    }
  }

  return std::shared_ptr<quickrank::optimization::Optimization>(optimizer);
}

std::shared_ptr<quickrank::optimization::Optimization> create_pruner(
    Cleaver::PruningMethod pruningMethod, double pruning_rate,
    std::shared_ptr<learning::linear::LineSearch> lineSearch) {

  Optimization *optimizer = nullptr;

  switch (pruningMethod) {
    case Cleaver::PruningMethod::RANDOM: {
      optimizer = new RandomPruning(pruning_rate, lineSearch);
      break;
    }
    case Cleaver::PruningMethod::RANDOM_ADV: {
      optimizer = new RandomAdvPruning(pruning_rate, lineSearch);
      break;
    }
    case Cleaver::PruningMethod::LOW_WEIGHTS: {
      optimizer = new LowWeightsPruning(pruning_rate, lineSearch);
      break;
    }
    case Cleaver::PruningMethod::LAST: {
      optimizer = new LastPruning(pruning_rate, lineSearch);
      break;
    }
    case Cleaver::PruningMethod::QUALITY_LOSS: {
      optimizer = new QualityLossPruning(pruning_rate, lineSearch);
      break;
    }
    case Cleaver::PruningMethod::QUALITY_LOSS_ADV: {
      optimizer = new QualityLossAdvPruning(pruning_rate, lineSearch);
      break;
    }
    case Cleaver::PruningMethod::SKIP: {
      optimizer = new SkipPruning(pruning_rate, lineSearch);
      break;
    }
    case Cleaver::PruningMethod::SCORE_LOSS: {
      optimizer = new ScoreLossPruning(pruning_rate, lineSearch);
      break;
    }
  }

  return std::shared_ptr<quickrank::optimization::Optimization>(optimizer);
}

std::shared_ptr<quickrank::optimization::Optimization> create_pruner(
    Cleaver::PruningMethod pruningMethod, double pruning_rate) {

  return create_pruner(pruningMethod, pruning_rate, nullptr);
}

std::shared_ptr<quickrank::optimization::Optimization> create_pruner(
    std::string pruningMethodName, double pruning_rate,
    std::shared_ptr<learning::linear::LineSearch> lineSearch) {

  return create_pruner(
      Cleaver::get_pruning_method(pruningMethodName),
      pruning_rate, lineSearch);

}

std::shared_ptr<quickrank::optimization::Optimization> create_pruner(
    std::string pruningMethodName, double pruning_rate) {

  return create_pruner(pruningMethodName, pruning_rate, nullptr);
}

}
}
}
}