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

#include "pugixml/src/pugixml.hpp"
#include "learning/ltr_algorithm.h"

#include "learning/forests/mart.h"
#include "learning/forests/dart.h"
#include "learning/forests/lambdamart.h"
#include "learning/forests/obliviouslambdamart.h"
#include "learning/forests/obliviousmart.h"
// Added by Chiara Pierucci Andrea Battistini
#include "learning/linear/coordinate_ascent.h"
// Added by Tommaso Papini and Gabriele Bani
#include "learning/forests/rankboost.h"
// Added by Salvatore Trani
#include "learning/linear/line_search.h"
#include "optimization/post_learning/cleaver/cleaver.h"

namespace quickrank {
namespace learning {

void LTR_Algorithm::score_dataset(std::shared_ptr<data::Dataset> dataset,
                                  Score *scores) const {
  const quickrank::Feature *d = dataset->at(0, 0);
  #pragma omp parallel for
  for (size_t i = 0; i < dataset->num_instances(); i++) {
    scores[i] = score_document(d + i * dataset->num_features());
//    d += dataset->num_features();
  }
}

void LTR_Algorithm::save(std::string output_basename, int iteration) const {
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

std::shared_ptr<LTR_Algorithm> LTR_Algorithm::load_model_from_file(
    std::string model_filename) {
  if (model_filename.empty()) {
    std::cerr << "!!! Model filename is empty." << std::endl;
    exit(EXIT_FAILURE);
  }

  pugi::xml_document model;
  pugi::xml_parse_result result = model.load_file(model_filename.c_str());
  if (!result) {
    std::cerr << "!!! Model " + model_filename + " is not parsed correctly."
              << std::endl;
    exit(EXIT_FAILURE);
  }

  return load_model_from_xml(model);
}

std::shared_ptr<LTR_Algorithm> LTR_Algorithm::load_model_from_xml(
    const pugi::xml_document& xml_model) {

  std::string ranker_type =
      xml_model.child("ranker").child("info").child("type").child_value();

  if (ranker_type == forests::Mart::NAME_)
    return std::shared_ptr<LTR_Algorithm>(
        new forests::Mart(xml_model));
  else if (ranker_type == forests::Dart::NAME_)
    return std::shared_ptr<LTR_Algorithm>(
        new forests::Dart(xml_model));
  else if (ranker_type == forests::LambdaMart::NAME_)
    return std::shared_ptr<LTR_Algorithm>(
        new forests::LambdaMart(xml_model));
  else if (ranker_type == forests::ObliviousMart::NAME_)
    return std::shared_ptr<LTR_Algorithm>(
        new forests::ObliviousMart(xml_model));
  else if (ranker_type == forests::ObliviousLambdaMart::NAME_)
    return std::shared_ptr<LTR_Algorithm>(
        new forests::ObliviousLambdaMart(xml_model));
    // Coordinate Ascent added by Chiara Pierucci and Andrea Battistini
  else if (ranker_type == linear::CoordinateAscent::NAME_)
    return std::shared_ptr<LTR_Algorithm>(
        new linear::CoordinateAscent(xml_model));
    // Rankboost added by Tommaso Papini and Gabriele Bani
  else if (ranker_type == forests::Rankboost::NAME_)
    return std::shared_ptr<LTR_Algorithm>(
        new forests::Rankboost(xml_model));
    // Line Search added by Salvatore Trani
  else if (ranker_type == linear::LineSearch::NAME_)
    return std::shared_ptr<LTR_Algorithm>(
        new linear::LineSearch(xml_model));
  else if (ranker_type == meta::MetaCleaver::NAME_)
    return std::shared_ptr<LTR_Algorithm>(
        new meta::MetaCleaver(xml_model));

  return nullptr;
  //  else
  //    throw std::invalid_argument("Model type not supported for loading");
}

}  // namespace learning
}  // namespace quickrank
