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
#include <boost/property_tree/xml_parser.hpp>
#include <boost/property_tree/detail/ptree_utils.hpp>
#include <boost/foreach.hpp>
#include <boost/container/list.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/iterator/zip_iterator.hpp>
#include <boost/range.hpp>
#include <boost/algorithm/string.hpp>

#include <string>
#include <memory>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <limits>

#include "io/xml.h"

#include "learning/forests/mart.h"
#include "learning/forests/lambdamart.h"
#include "learning/forests/obliviouslambdamart.h"

namespace quickrank {
namespace io {

RTNode* RTNode_parse_xml(const boost::property_tree::ptree &split_xml) {
  RTNode* model_node = NULL;
  RTNode* left_child = NULL;
  RTNode* right_child = NULL;

  bool is_leaf = false;

  unsigned int feature_id = 0;
  Feature threshold = 0.0f;
  Score prediction = 0.0;

  BOOST_FOREACH(const boost::property_tree::ptree::value_type& split_child, split_xml ){
  if (split_child.first == "output") {
    prediction = split_child.second.get_value<Score>();
    is_leaf = true;
    break;
  } else if (split_child.first == "feature") {
    feature_id = split_child.second.get_value<unsigned int>();
  } else if (split_child.first == "threshold") {
    threshold = split_child.second.get_value<Feature>();
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
  std::string threshold;
  std::string prediction;
  bool is_leaf = false;
  const boost::property_tree::ptree* left = NULL;
  const boost::property_tree::ptree* right = NULL;

  BOOST_FOREACH(const boost::property_tree::ptree::value_type& split_child, split_xml ){
  if (split_child.first == "output") {
    prediction = split_child.second.get_value<std::string>();
    boost::algorithm::trim(prediction);
    is_leaf = true;
    break;
  } else if (split_child.first == "feature") {
    feature_id = split_child.second.get_value<unsigned int>();
  } else if (split_child.first == "threshold") {
    threshold = split_child.second.get_value<std::string>();
    boost::algorithm::trim(threshold);
  } else if (split_child.first == "split") {
    std::string pos = split_child.second.get<std::string>("<xmlattr>.pos");
    if (pos == "left")
    left = &(split_child.second);
    else
    right = &(split_child.second);
  }
}

  if (is_leaf)
    os << prediction;
  else {
    /// \todo TODO: this should be changed with item mapping
    os << "( v[" << feature_id - 1 << "] <= ";
    os << threshold << "f";
    os << " ? ";
    model_node_to_c_baseline(*left, os);
    os << " : ";
    model_node_to_c_baseline(*right, os);
    os << " )";
  }
}

void model_node_to_c_oblivious_trees(
    const boost::property_tree::ptree &split_xml, std::stringstream &os) {

  std::string prediction;
  bool is_leaf = false;
  const boost::property_tree::ptree* left = NULL;
  const boost::property_tree::ptree* right = NULL;

  BOOST_FOREACH(const boost::property_tree::ptree::value_type& split_child, split_xml ){
  if (split_child.first == "output") {
    prediction = split_child.second.get_value<std::string>();
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
    os << prediction;
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
  unsigned int feature = 0;
  float threshold = 0.0f;
  bool is_leaf = false;
  for (auto p_node = tree_xml.begin(); p_node != tree_xml.end(); p_node++) {
    if (p_node->first == "split") {
      std::string pos = p_node->second.get < std::string > ("<xmlattr>.pos");
      if (pos == "left")
        left = &(p_node->second);
      else
        right = &(p_node->second);
    } else if (p_node->first == "feature") {
      feature = p_node->second.get_value<unsigned int>();
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

  namespace bc = boost::container;
  bc::list<bc::list<unsigned int> > tree_features;
  bc::list<bc::list<float> > tree_thresholds;

// collect data
  auto ensemble = xml_model.get_child("ranker.ensemble");
  for (auto p_tree = ensemble.begin(); p_tree != ensemble.end(); ++p_tree) {
    boost::container::list<unsigned int> features;
    boost::container::list<float> thresholds;
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
  unsigned int t = 0;
  for (; features != tree_features.end() && thresholds != tree_thresholds.end();
      ++features, ++thresholds, ++t) {
    // iterate over nodes of the given tree
    source_code << std::endl << "    ";
    if (t != 0)
      source_code << " + ";
    source_code << "(double)( ";
    auto f = features->begin();
    auto t = thresholds->begin();
    unsigned int l = 0;
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
  source_code.setf(std::ios::floatfield, std::ios::fixed);

  source_code << "double ranker(float* v) {" << std::endl;
  source_code << "\treturn 0.0 ";
  BOOST_FOREACH(const boost::property_tree::ptree::value_type& tree, xml_tree.get_child("ranker.ensemble")){
  float tree_weight = tree.second.get("<xmlattr>.weight", 1.0f);

// find the root of the tree
  boost::property_tree::ptree root;
  BOOST_FOREACH(const boost::property_tree::ptree::value_type& node, tree.second ) {
    if (node.first == "split") {
      source_code << std::endl << "\t\t + " << std::setprecision(3)
      << tree_weight << "f * ";
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
  source_code.setf(std::ios::floatfield, std::ios::fixed);
  source_code << "float ws[N] = { ";
  for (auto p_tree = ensemble.begin(); p_tree != ensemble.end(); p_tree++) {
    if (p_tree != ensemble.begin())
      source_code << ", ";
    float tree_weight = p_tree->second.get("<xmlattr>.weight", 1.0f);
    source_code << tree_weight << "f";
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
  source_code << std::setprecision(std::numeric_limits<Feature>::digits10);

  for (auto p_tree = ensemble.begin(); p_tree != ensemble.end(); p_tree++) {
    if (p_tree != ensemble.begin())
      source_code << "," << std::endl;

    source_code << "\t{ ";
    auto p_split = p_tree->second.get_child("split");
    std::string separator = "";
    while (p_split.size() != 2) {
      std::string threshold = p_split.get<std::string>("threshold");
      boost::algorithm::trim(threshold);
      source_code << separator << threshold << "f";
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
      << "  for (unsigned int i=0; i<m; ++i)" << std::endl
      << "    leafidx |= SHL( v[fids[i]-1]>thresholds[i], m-1-i);" << std::endl
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

  BOOST_FOREACH(const boost::property_tree::ptree::value_type& node, xml_tree.get_child("ranker")){
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
        new learning::forests::ObliviousLambdaMart(info_ptree, ensemble_ptree));

  return NULL;
}

}  // namespace io
}  // namespace quickrank
