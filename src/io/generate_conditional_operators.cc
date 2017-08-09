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

#include "io/generate_conditional_operators.h"

namespace quickrank {
namespace io {

void model_node_to_conditional_operators(pugi::xml_node &nodes,
                                         std::stringstream &os) {
  unsigned int feature_id = 0;
  std::string threshold;
  std::string prediction;
  bool is_leaf = false;
  pugi::xml_node left;
  pugi::xml_node right;

  for (pugi::xml_node &node : nodes.children()) {
    if (strcmp(node.name(), "output") == 0) {
      prediction = node.text().as_string();
      trim(prediction);
      is_leaf = true;
      break;
    } else if (strcmp(node.name(), "feature") == 0) {
      feature_id = node.text().as_uint();
    } else if (strcmp(node.name(), "threshold") == 0) {
      threshold = node.text().as_string();
      trim(threshold);
      // dealing with integer values
      if (threshold.find(".") == std::string::npos)
      // adding ".0" to deal with the integer found
        threshold += ".0";
    } else if (strcmp(node.name(), "split") == 0) {
      std::string pos = node.attribute("pos").as_string();

      if (pos == "left") {
        left = node;
      }
      if (pos == "right") {
        right = node;
      }
    }
  }

  if (is_leaf)
    os << prediction;
  else {
    /// \todo TODO: this should be changed with item mapping
    os << "( v[" << feature_id - 1 << "] <= ";
    os << threshold << "f";
    os << " ? ";
    model_node_to_conditional_operators(left, os);
    os << " : ";
    model_node_to_conditional_operators(right, os);
    os << " )";
  }
}

void
GenOpCond::generate_conditional_operators_code(const std::string model_filename,
                                               const std::string code_filename) {
  if (model_filename.empty()) {
    std::cerr << "!!! Model filename is empty." << std::endl;
    exit(EXIT_FAILURE);
  }

  // loading XML
  pugi::xml_document xml_document;
  xml_document.load_file(model_filename.c_str());

  // defining output string
  std::stringstream source_code;
  source_code.setf(std::ios::floatfield, std::ios::fixed);

  // printing header
  source_code << "double ranker(float* v) {" << std::endl;
  source_code << "\treturn 0.0 ";

  // let's navigate the ensemble, for each tree...
  pugi::xml_node ensemble = xml_document.child("ranker").child("ensemble");
  for (pugi::xml_node &tree : ensemble.children("tree")) {
    float tree_weight = tree.attribute("weight").as_float();
    pugi::xml_node tree_content = tree.child("split");
    if (tree_content) {
      source_code << std::endl << "\t\t + " << std::setprecision(3)
                  << tree_weight << "f * ";
      model_node_to_conditional_operators(tree_content, source_code);
    }
  }
  source_code << ";" << std::endl << "}" << std::endl;

  std::ofstream output;
  output.open(code_filename, std::ofstream::out);
  output << source_code.str();
  output.close();
}

}  // namespace io
}  // namespace quickrank
