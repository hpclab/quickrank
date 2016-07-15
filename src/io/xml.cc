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
#include <string>
#include <memory>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <limits>
#include <list>
#include <stdlib.h>

#include "pugixml/src/pugixml.hpp"

#include "io/xml.h"
#include "utils/strutils.h"

#include "learning/forests/mart.h"
#include "learning/forests/lambdamart.h"
#include "learning/forests/obliviouslambdamart.h"

namespace quickrank {
namespace io {

/*
void model_tree_get_leaves(const boost::property_tree::ptree &split_xml,
                           std::vector<std::string> &leaves) {
  std::string prediction;
  bool is_leaf = false;
  const boost::property_tree::ptree* left = NULL;
  const boost::property_tree::ptree* right = NULL;

  for (const boost::property_tree::ptree::value_type& split_child: split_xml) {
    if (split_child.first == "output") {
      prediction = split_child.second.get_value<std::string>();
      trim(prediction);
      is_leaf = true;
      break;
    } else if (split_child.first == "split") {
      std::string pos = split_child.second.get<std::string>("<xmlattr>.pos");
      if (pos == "left")
        left = &(split_child.second);
      else
        right = &(split_child.second);
    }
  }

  if (is_leaf)
    leaves.push_back(prediction);
  else {
    model_tree_get_leaves(*left, leaves);
    model_tree_get_leaves(*right, leaves);
  }
}

void model_tree_get_tests(const boost::property_tree::ptree &tree_xml,
                          std::list<size_t> &features,
                          std::list<float> &thresholds) {
  const boost::property_tree::ptree* left = NULL;
  const boost::property_tree::ptree* right = NULL;
  size_t feature = 0;
  float threshold = 0.0f;
  bool is_leaf = false;
  for (auto p_node = tree_xml.begin(); p_node != tree_xml.end(); p_node++) {
    if (p_node->first == "split") {
      std::string pos = p_node->second.get<std::string>("<xmlattr>.pos");
      if (pos == "left")
        left = &(p_node->second);
      else
        right = &(p_node->second);
    } else if (p_node->first == "feature") {
      feature = p_node->second.get_value<size_t>();
    } else if (p_node->first == "threshold") {
      threshold = p_node->second.get_value<float>();
    } else if (p_node->first == "output") {
      is_leaf = true;
    }
  }
  if (is_leaf)
    return;
  model_tree_get_tests(*left, features, thresholds);
  features.push_back(feature);
  thresholds.push_back(threshold);
  model_tree_get_tests(*right, features, thresholds);
}

void Xml::generate_c_code_vectorized(std::string model_filename,
                                     std::string code_filename) {
  if (model_filename.empty()) {
    std::cerr << "!!! Model filename is empty." << std::endl;
    exit(EXIT_FAILURE);
  }

// parse XML
  boost::property_tree::ptree xml_model;
  std::ifstream is;
  is.open(model_filename, std::ifstream::in);
  boost::property_tree::read_xml(is, xml_model);
  is.close();

  std::list<std::list<size_t> > tree_features;
  std::list<std::list<float> > tree_thresholds;

// collect data
  auto ensemble = xml_model.get_child("ranker.ensemble");
  for (auto p_tree = ensemble.begin(); p_tree != ensemble.end(); ++p_tree) {
    std::list<size_t> features;
    std::list<float> thresholds;
    auto tree_root = p_tree->second.find("split");
    model_tree_get_tests(tree_root->second, features, thresholds);
    tree_features.push_back(features);
    tree_thresholds.push_back(thresholds);
  }

// create output stream
  std::stringstream source_code;
  source_code << "double ranker(float *v) {" << std::endl << "  double score =";

// iterate over trees
  auto features = tree_features.begin();
  auto thresholds = tree_thresholds.begin();
  size_t t = 0;
  for (; features != tree_features.end() && thresholds != tree_thresholds.end();
      ++features, ++thresholds, ++t) {
    // iterate over nodes of the given tree
    source_code << std::endl << "    ";
    if (t != 0)
      source_code << " + ";
    source_code << "(double)( ";
    auto f = features->begin();
    auto t = thresholds->begin();
    size_t l = 0;
    for (; f != features->end() && t != thresholds->end(); ++f, ++t, ++l) {
      if (l != 0)
        source_code << " | ";
      source_code << "((v[" << *f << "]<=" << *t << ")<<" << l << ")";
    }
    source_code << " )";
  }
  source_code << ";" << std::endl;
  source_code << "  return score;" << std::endl << "}" << std::endl
              << std::endl;

  std::ofstream output;
  output.open(code_filename, std::ofstream::out);
  output << source_code.str();
  output.close();

}
*/
}  // namespace io
}  // namespace quickrank
