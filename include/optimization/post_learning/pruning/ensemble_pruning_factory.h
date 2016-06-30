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
#pragma once

#include "optimization/optimization.h"
#include "optimization/post_learning/pruning/ensemble_pruning.h"

#include "optimization/post_learning/pruning/RandomPruning.h"
#include "optimization/post_learning/pruning/LastPruning.h"
#include "optimization/post_learning/pruning/SkipPruning.h"
#include "optimization/post_learning/pruning/ScoreLossPruning.h"
#include "optimization/post_learning/pruning/LowWeightsPruning.h"
#include "optimization/post_learning/pruning/QualityLossPruning.h"

namespace quickrank {
namespace optimization {
namespace post_learning {
namespace pruning {

  std::shared_ptr<quickrank::optimization::Optimization> create_pruner(
      EnsemblePruning::PruningMethod pruningMethod, double pruning_rate,
      std::shared_ptr<learning::linear::LineSearch> lineSearch) {

    Optimization *optimizer = nullptr;

    switch (pruningMethod) {
      case EnsemblePruning::PruningMethod::RANDOM: {
        optimizer = new RandomPruning(pruning_rate, lineSearch);
        break;
      }
      case EnsemblePruning::PruningMethod::LOW_WEIGHTS: {
        optimizer = new LowWeightsPruning(pruning_rate, lineSearch);
        break;
      }
      case EnsemblePruning::PruningMethod::LAST: {
        optimizer = new LastPruning(pruning_rate, lineSearch);
        break;
      }
      case EnsemblePruning::PruningMethod::QUALITY_LOSS: {
        optimizer = new QualityLossPruning(pruning_rate, lineSearch);
        break;
      }
      case EnsemblePruning::PruningMethod::SKIP: {
        optimizer = new SkipPruning(pruning_rate, lineSearch);
        break;
      }
      case EnsemblePruning::PruningMethod::SCORE_LOSS: {
        optimizer = new ScoreLossPruning(pruning_rate, lineSearch);
        break;
      }
    }

    return std::shared_ptr<quickrank::optimization::Optimization>(optimizer);
  }

  std::shared_ptr<quickrank::optimization::Optimization> create_pruner(
      EnsemblePruning::PruningMethod pruningMethod, double pruning_rate) {

    return create_pruner(pruningMethod, pruning_rate, nullptr);
  }

  std::shared_ptr<quickrank::optimization::Optimization> create_pruner(
      std::string pruningMethodName, double pruning_rate,
      std::shared_ptr<learning::linear::LineSearch> lineSearch) {

    return create_pruner(
        EnsemblePruning::getPruningMethod(pruningMethodName),
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