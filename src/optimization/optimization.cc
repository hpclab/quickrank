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

#include <boost/property_tree/xml_parser.hpp>
#include <boost/foreach.hpp>

#include "learning/ltr_algorithm.h"
#include "optimization/optimization.h"

namespace quickrank {
namespace optimization {

const std::vector<std::string> Optimization::optimizationAlgorithmNames = {
    "EPRUNING"
};

void Optimization::save(std::string output_basename, int iteration) const {
  if (!output_basename.empty()) {
    std::string filename(output_basename);
    if (iteration != -1)
      filename += ".T" + std::to_string(iteration) + ".xml";
    std::ofstream output_stream;
    output_stream.open(filename);
    // Wrap actual model
    output_stream << "<optimizer>" << std::endl;

    // Save the actual model
    save_model_to_file(output_stream);

    output_stream << "</optimizer>" << std::endl;

    output_stream.close();
  }
}

std::shared_ptr<Optimization> Optimization::load_model_from_file(
    std::string model_filename) {

  if (model_filename.empty()) {
    std::cerr << "!!! Model filename is empty." << std::endl;
    exit(EXIT_FAILURE);
  }

  boost::property_tree::ptree xml_tree;

  std::ifstream is;
  is.open(model_filename, std::ifstream::in);

  boost::property_tree::read_xml(is, xml_tree);

  is.close();

  boost::property_tree::ptree info_ptree;
  boost::property_tree::ptree model_ptree;

  for (const boost::property_tree::ptree::value_type& node:
                            xml_tree.get_child("optimizer")) {
    if (node.first == "info")
      info_ptree = node.second;
    else
      model_ptree = node.second;
  }

  std::string optimizer_type = info_ptree.get<std::string>("type");
    // Ensemble Pruning added by Salvatore Trani
//  if (optimizer_type == optimization::post_learning::pruning::EnsemblePruning::NAME_)
//    return std::shared_ptr<Optimization>(
//        new pruning::EnsemblePruning(info_ptree, model_ptree));

  return nullptr;
//  else
//    throw std::invalid_argument("Model type not supported for loading");
  }

}  // namespace optimization
}  // namespace quickrank
