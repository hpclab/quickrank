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
#include <fstream>

#include "learning/ltr_algorithm.h"
#include "optimization/optimization.h"
#include "optimization/post_learning/cleaver/cleaver.h"
#include "optimization/post_learning/cleaver/cleaver_factory.h"

namespace quickrank {
namespace optimization {

const std::vector<std::string> Optimization::optimizationAlgorithmNames = {
    "CLEAVER"
};

void Optimization::save(std::string output_basename, int iteration) const {
  if (!output_basename.empty()) {
    std::string filename(output_basename);
    if (iteration != -1)
      filename += ".T" + std::to_string(iteration) + ".xml";

    pugi::xml_document *doc = get_xml_model();
    doc->save_file(filename.c_str(), "\t",
                   pugi::format_default | pugi::format_no_declaration);
    delete (doc);
  }
}

std::shared_ptr<Optimization> Optimization::load_model_from_file(
    std::string model_filename) {

  if (model_filename.empty()) {
    std::cerr << "!!! Model filename is empty." << std::endl;
    exit(EXIT_FAILURE);
  }

  pugi::xml_document model;
  pugi::xml_parse_result result = model.load_file(model_filename.c_str());
  if (!result) {
    std::cerr << "!!! Model filename is not parsed correctly." << std::endl;
    exit(EXIT_FAILURE);
  }

  std::string optimizer_type =
      model.child("optimizer").child("info").child("opt-algo").child_value();

  // Ensemble Pruning added by Salvatore Trani
  if (optimizer_type
      == optimization::post_learning::pruning::Cleaver::NAME_)
    return optimization::post_learning::pruning::create_pruner(model);

  return nullptr;
//  else
//    throw std::invalid_argument("Model type not supported for loading");
}

}  // namespace optimization
}  // namespace quickrank
