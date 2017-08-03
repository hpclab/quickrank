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

#include "pugixml/src/pugixml.hpp"

#include "optimization/optimization.h"
#include "learning/linear/line_search.h"
#include "optimization/post_learning/cleaver/cleaver.h"

#include <memory>

namespace quickrank {
namespace optimization {
namespace post_learning {
namespace pruning {

std::shared_ptr<quickrank::optimization::Optimization> create_pruner(
    const pugi::xml_document &model);

std::shared_ptr<quickrank::optimization::Optimization> create_pruner(
    Cleaver::PruningMethod pruningMethod, double pruning_rate,
    std::shared_ptr<learning::linear::LineSearch> lineSearch);

std::shared_ptr<quickrank::optimization::Optimization> create_pruner(
    Cleaver::PruningMethod pruningMethod, double pruning_rate);

std::shared_ptr<quickrank::optimization::Optimization> create_pruner(
    std::string pruningMethodName, double pruning_rate,
    std::shared_ptr<learning::linear::LineSearch> lineSearch);

std::shared_ptr<quickrank::optimization::Optimization> create_pruner(
    std::string pruningMethodName, double pruning_rate);

}
}
}
}
