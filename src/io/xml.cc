/*
 * QuickRank - A C++ suite of Learning to Rank algorithms
 * Webpage: http://quickrank.isti.cnr.it/
 * Contact: quickrank@isti.cnr.it
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 *
 * Contributor:
 *   HPC. Laboratory - ISTI - CNR - http://hpc.isti.cnr.it/
 */
#include <boost/property_tree/xml_parser.hpp>
#include <boost/property_tree/detail/ptree_utils.hpp>
#include <boost/foreach.hpp>
#include <boost/container/list.hpp>
#include <boost/lexical_cast.hpp>

#include <string>
#include <memory>
#include <fstream>
#include <sstream>

#include "io/xml.h"

#include "learning/forests/mart.h"
#include "learning/forests/lambdamart.h"
#include "learning/forests/matrixnet.h"

namespace quickrank {
namespace io {

RTNode* RTNode_parse_xml(const boost::property_tree::ptree &split_xml) {
  RTNode* model_node = NULL;
  RTNode* left_child = NULL;
  RTNode* right_child = NULL;

  bool is_leaf = false;

  unsigned int feature_id = 0;
  float threshold = 0.0f;
  double prediction = 0.0;

  BOOST_FOREACH(const boost::property_tree::ptree::value_type& split_child, split_xml ) {
    if (split_child.first == "output") {
      prediction = split_child.second.get_value<double>();
      is_leaf = true;
      break;
    } else if (split_child.first == "feature") {
      feature_id = split_child.second.get_value<unsigned int>();
    } else if (split_child.first == "threshold") {
      threshold = split_child.second.get_value<float>();
    } else if (split_child.first == "split") {
      std::string pos = split_child.second.get<std::string>("<xmlattr>.pos");
      if (pos == "left")
        left_child = RTNode_parse_xml(split_child.second);
      else
        right_child = RTNode_parse_xml(split_child.second);
    }
  }

  if (is_leaf)
    model_node = new RTNode(prediction);
  else
    /// \todo TODO: this should be changed with item mapping
    model_node = new RTNode(threshold, feature_id - 1, feature_id, left_child,
                            right_child);

  return model_node;
}

void model_node_to_c_baseline(const boost::property_tree::ptree &split_xml,
                              std::stringstream &os) {
  unsigned int feature_id = 0;
  float threshold = 0.0f;
  double prediction = 0.0;
  bool is_leaf = false;
  const boost::property_tree::ptree* left = NULL;
  const boost::property_tree::ptree* right = NULL;

  BOOST_FOREACH(const boost::property_tree::ptree::value_type& split_child, split_xml ) {
    if (split_child.first == "output") {
      prediction = split_child.second.get_value<double>();
      is_leaf = true;
      break;
    } else if (split_child.first == "feature") {
      feature_id = split_child.second.get_value<unsigned int>();
    } else if (split_child.first == "threshold") {
      threshold = split_child.second.get_value<float>();
    } else if (split_child.first == "split") {
      std::string pos = split_child.second.get<std::string>("<xmlattr>.pos");
      if (pos == "left")
        left = &(split_child.second);
      else
        right = &(split_child.second);
    }
  }

  if (is_leaf)
    os << std::setprecision(15) << prediction;
  else {
    /// \todo TODO: this should be changed with item mapping
    os << "( v[" << feature_id - 1 << "] <= " << std::setprecision(15)
       << threshold;
    os << " ? ";
    model_node_to_c_baseline(*left, os);
    os << " : ";
    model_node_to_c_baseline(*right, os);
    os << " )";
  }
}

void model_node_to_c_oblivious_trees(
    const boost::property_tree::ptree &split_xml, std::stringstream &os) {

  double prediction = 0.0;
  bool is_leaf = false;
  const boost::property_tree::ptree* left = NULL;
  const boost::property_tree::ptree* right = NULL;

  BOOST_FOREACH(const boost::property_tree::ptree::value_type& split_child, split_xml ) {
    if (split_child.first == "output") {
      prediction = split_child.second.get_value<double>();
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
    os << std::setprecision(15) << prediction;
  else {
    /// \todo TODO: this should be changed with item mapping
    //os << " ";
    model_node_to_c_oblivious_trees(*left, os);
    os << ", ";
    model_node_to_c_oblivious_trees(*right, os);
    //os << " ";
  }
}


void model_tree_get_tests(const boost::property_tree::ptree &tree_xml,
                          boost::container::list<unsigned int> &features,
                          boost::container::list<float> &thresholds) {
  const boost::property_tree::ptree* left = NULL;
  const boost::property_tree::ptree* right = NULL;
  unsigned int feature;
  float threshold;
  bool is_leaf = false;
  for (auto p_node = tree_xml.begin(); p_node != tree_xml.end(); p_node++) {
    if (p_node->first=="split") {
      std::string pos = p_node->second.get<std::string>("<xmlattr>.pos");
      if (pos=="left")
        left = &(p_node->second);
      else
        right = &(p_node->second);
    } else if (p_node->first=="feature") {
      feature = p_node->second.get_value<unsigned int>();
    } else if (p_node->first=="threshold") {
      threshold = p_node->second.get_value<float>();
    } else if (p_node->first=="output") {
      is_leaf = true;
    }
  }
  if (is_leaf) return;
  model_tree_get_tests(*left, features, thresholds);
  std::cout << feature << "<=" << threshold << std::endl;
  model_tree_get_tests(*right, features, thresholds);
}

void model_get_tests(const boost::property_tree::ptree &model_xml) {
  auto ensemble = model_xml.get_child("ranker.ensemble");
  for (auto p_tree = ensemble.begin(); p_tree != ensemble.end(); p_tree++) {
    boost::container::list<unsigned int> features;
    boost::container::list<float> thresholds;
    auto tree_root = p_tree->second.find("split");
    model_tree_get_tests(tree_root->second, features, thresholds);
  }
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

  model_get_tests(xml_model);
}

void Xml::generate_c_code_baseline(std::string model_filename,
                                   std::string code_filename) {
  if (model_filename.empty()) {
    std::cerr << "!!! Model filename is empty." << std::endl;
    exit(EXIT_FAILURE);
  }
  // parse XML
  boost::property_tree::ptree xml_tree;
  std::ifstream is;
  is.open(model_filename, std::ifstream::in);
  boost::property_tree::read_xml(is, xml_tree);
  is.close();

  // create output stream
  std::stringstream source_code;

  source_code << "double ranker(float* v) {" << std::endl;
  source_code << "\treturn 0.0 ";
  BOOST_FOREACH(const boost::property_tree::ptree::value_type& tree, xml_tree.get_child("ranker.ensemble")) {
    float tree_weight = tree.second.get("<xmlattr>.weight", 1.0f);

    // find the root of the tree
    boost::property_tree::ptree root;
    BOOST_FOREACH(const boost::property_tree::ptree::value_type& node, tree.second ) {
      if (node.first == "split") {
        source_code << std::endl << "\t\t + " << std::setprecision(3)
                    << tree_weight << " * ";
        model_node_to_c_baseline(node.second, source_code);
      }
    }
  }
  source_code << ";" << std::endl << "}" << std::endl;

  std::ofstream output;
  output.open(code_filename, std::ofstream::out);
  output << source_code.str();
  output.close();
}

void Xml::generate_c_code_oblivious_trees(std::string model_filename,
                                          std::string code_filename) {
  if (model_filename.empty()) {
    std::cerr << "!!! Model filename is empty." << std::endl;
    exit(EXIT_FAILURE);
  }

  // parse XML
  boost::property_tree::ptree xml_tree;
  std::ifstream is;
  is.open(model_filename, std::ifstream::in);
  boost::property_tree::read_xml(is, xml_tree);
  is.close();

  // create output stream
  std::stringstream source_code;

  auto ensemble = xml_tree.get_child("ranker.ensemble");
  unsigned int trees = ensemble.size();
  unsigned int depth = xml_tree.get<unsigned int>("ranker.info.depth");

  // Forests info
  source_code << "#define N " << trees << " // no. of trees" << std::endl;
  source_code << "#define M " << depth << " // max tree depth" << std::endl;
  source_code << std::endl;

  // Tree Weights
  source_code << "double ws[N] = { ";
  for (auto p_tree = ensemble.begin(); p_tree != ensemble.end(); p_tree++) {
    if (p_tree != ensemble.begin())
      source_code << ", ";
    float tree_weight = p_tree->second.get("<xmlattr>.weight", 1.0f);
    source_code << tree_weight;
  }
  source_code << " };" << std::endl << std::endl;

  // Actual Tree Depths
  int counter;
  source_code << "unsigned int ds[N] = { ";
  for (auto p_tree = ensemble.begin(); p_tree != ensemble.end(); p_tree++) {
    if (p_tree != ensemble.begin())
      source_code << ", ";

    counter = 0;
    auto p_split = p_tree->second.get_child("split");
    while (p_split.size() != 2) {
      counter++;
      p_split = p_split.get_child("split");
    }
    source_code << counter;
  }
  source_code << " };" << std::endl << std::endl;

  // Leaf Outputs
  source_code << "double os[N][1 << M] = { " << std::endl;
  for (auto p_tree = ensemble.begin(); p_tree != ensemble.end(); p_tree++) {
    if (p_tree != ensemble.begin())
      source_code << "," << std::endl;
    auto root_split = p_tree->second.get_child("split");
    source_code << "\t{ ";
    model_node_to_c_oblivious_trees(root_split, source_code);
    source_code << "}";
  }
  source_code << std::endl << "};" << std::endl << std::endl;

  // Features ids
  source_code << "unsigned int fs[N][M] = { " << std::endl;
  for (auto p_tree = ensemble.begin(); p_tree != ensemble.end(); p_tree++) {
    if (p_tree != ensemble.begin())
      source_code << "," << std::endl;

    source_code << "\t{ ";
    auto p_split = p_tree->second.get_child("split");
    std::string separator = "";
    while (p_split.size() != 2) {
      source_code << separator << p_split.get<unsigned int>("feature");
      p_split = p_split.get_child("split");
      separator = ", ";
    }
    source_code << "}";
  }
  source_code << std::endl << "};" << std::endl << std::endl;

  // Thresholds values
  source_code << "float ts[N][M] = { " << std::endl;
  for (auto p_tree = ensemble.begin(); p_tree != ensemble.end(); p_tree++) {
    if (p_tree != ensemble.begin())
      source_code << "," << std::endl;

    source_code << "\t{ ";
    auto p_split = p_tree->second.get_child("split");
    std::string separator = "";
    while (p_split.size() != 2) {
      source_code << separator << p_split.get<float>("threshold");
      p_split = p_split.get_child("split");
      separator = ", ";
    }
    source_code << "}";
  }
  source_code << std::endl << "};" << std::endl << std::endl;

  source_code << "#define SHL(n,p) ((n)<<(p))" << std::endl << std::endl;

  source_code
      << "unsigned int leaf_id(float *v, unsigned int const *fids, float const *thresholds, const unsigned int m) {"
      << std::endl << "  unsigned int leafidx = 0;" << std::endl
      << "  for (unsigned int i = 0; i<M && i < m; ++i)" << std::endl
      << "    leafidx |= SHL( v[fids[i]-1]>thresholds[i], M-1-i);" << std::endl
      << "  return leafidx;" << std::endl << "}" << std::endl << std::endl;

  source_code << "double ranker(float *v) {" << std::endl
              << "  double score = 0.0;" << std::endl
              << "  //#pragma omp parallel for reduction(+:score)" << std::endl
              << "  for (int i = 0; i < N; ++i)" << std::endl
              << "    score += ws[i] * os[i][leaf_id(v, fs[i], ts[i], ds[i])];"
              << std::endl << "  return score;" << std::endl << "}" << std::endl
              << std::endl;

  std::ofstream output;
  output.open(code_filename, std::ofstream::out);
  output << source_code.str();
  output.close();
}

std::shared_ptr<learning::LTR_Algorithm> Xml::load_model_from_file(
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
  boost::property_tree::ptree ensemble_ptree;

  BOOST_FOREACH(const boost::property_tree::ptree::value_type& node, xml_tree.get_child("ranker")) {
    if (node.first == "info")
      info_ptree = node.second;
    else if (node.first == "ensemble")
      ensemble_ptree = node.second;
  }

  std::string ranker_type = info_ptree.get<std::string>("type");
  if (ranker_type == "MART")
    return std::shared_ptr<learning::LTR_Algorithm>(
        new learning::forests::Mart(info_ptree, ensemble_ptree));
  if (ranker_type == "LAMBDAMART")
    return std::shared_ptr<learning::LTR_Algorithm>(
        new learning::forests::LambdaMart(info_ptree, ensemble_ptree));
  if (ranker_type == "MATRIXNET")
    return std::shared_ptr<learning::LTR_Algorithm>(
        new learning::forests::MatrixNet(info_ptree, ensemble_ptree));

  return NULL;
}

}  // namespace io
}  // namespace quickrank
