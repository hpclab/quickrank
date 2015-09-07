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

  BOOST_FOREACH(const boost::property_tree::ptree::value_type& split_child, split_xml ) {
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

  BOOST_FOREACH(const boost::property_tree::ptree::value_type& split_child, split_xml ) {
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



void model_tree_get_leaves(const boost::property_tree::ptree &split_xml, std::vector<std::string> &leaves) {
  std::string prediction;
  bool is_leaf = false;
  const boost::property_tree::ptree* left = NULL;
  const boost::property_tree::ptree* right = NULL;

  BOOST_FOREACH(const boost::property_tree::ptree::value_type& split_child, split_xml ) {
    if (split_child.first == "output") {
      prediction = split_child.second.get_value<std::string>();
      boost::algorithm::trim(prediction);
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
    leaves.push_back( prediction );
  else {
    model_tree_get_leaves(*left, leaves);
    model_tree_get_leaves(*right, leaves);
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
      std::string pos = p_node->second.get<std::string>("<xmlattr>.pos");
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
  BOOST_FOREACH(const boost::property_tree::ptree::value_type& tree, xml_tree.get_child("ranker.ensemble")) {
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
  unsigned int max_leaves = 1 << depth;

// forests info
  source_code << "#define N " << trees << " // no. of trees" << std::endl;
  source_code << "#define M " << depth << " // max tree depth" << std::endl;
  source_code << "#define F " << max_leaves << " // max number of leaves" << std::endl;
  source_code << std::endl;

  // load tree weights
  std::vector<float> tree_weights;
  for (auto p_tree = ensemble.begin(); p_tree != ensemble.end(); p_tree++)
     tree_weights.push_back( p_tree->second.get("<xmlattr>.weight", 1.0f) );

  // load tree depths
  std::vector<int> tree_depths;
  for (auto p_tree = ensemble.begin(); p_tree != ensemble.end(); p_tree++) {
    int curr_depth = 0;
    auto p_split = p_tree->second.get_child("split");
    while (p_split.size() != 2) {
      curr_depth++;
      p_split = p_split.get_child("split");
    }
    tree_depths.push_back(curr_depth);
  }

  // load leaf outputs
  std::vector< std::vector<std::string> > tree_outputs (trees);
  size_t curr_tree=0;
  for (auto p_tree = ensemble.begin(); p_tree != ensemble.end(); p_tree++) {
    auto root_split = p_tree->second.get_child("split");
    model_tree_get_leaves(root_split, tree_outputs[curr_tree++] );
  }

  // load features ids
  std::vector< std::vector<unsigned int> > feature_ids (trees);
  curr_tree=0;
  for (auto p_tree = ensemble.begin(); p_tree != ensemble.end(); p_tree++) {
    auto p_split = p_tree->second.get_child("split");
    while (p_split.size() != 2) {
      feature_ids[curr_tree].push_back( p_split.get<unsigned int>("feature") -1 );
      p_split = p_split.get_child("split");
    }
    curr_tree++;
  }

  // load thresholds values
  std::vector< std::vector<std::string> > thresholds (trees);
  curr_tree=0;
  for (auto p_tree = ensemble.begin(); p_tree != ensemble.end(); p_tree++) {
    auto p_split = p_tree->second.get_child("split");
    while (p_split.size() != 2) {
      std::string threshold = p_split.get<std::string>("threshold");
      boost::algorithm::trim(threshold);
      thresholds[curr_tree].push_back( threshold );
      p_split = p_split.get_child("split");
    }
    curr_tree++;
  }

  // mapping of trees in sorted order by depths
  std::vector<size_t> tree_mapping (trees);
  std::iota(tree_mapping.begin(), tree_mapping.end(), 0);
  std::sort( tree_mapping.begin(), tree_mapping.end(),
            [&tree_depths](int a, int b) {
                    return tree_depths[a] < tree_depths[b];
                }
  );

  // number of trees for each depth
  std::vector<size_t> depths_pupolation;
  int max_depth = tree_depths[ tree_mapping.back() ];

  int curr_depth = 1;
  size_t start_position = 0;
  for (size_t i=0; i<tree_mapping.size(); i++) {
    while ( tree_depths[ tree_mapping[i] ] > curr_depth ) {
      depths_pupolation.push_back( i-start_position );
      curr_depth++;
      start_position = i;
    }
    if (curr_depth==max_depth) break;
  }
  depths_pupolation.push_back( tree_mapping.size()-start_position );


  // print tree weights
  source_code.setf(std::ios::floatfield, std::ios::fixed);
  source_code << "const float tree_weights[N] = { ";
  for (size_t i = 0; i<tree_weights.size(); i++) {
    if (i!=0) source_code << ", ";
    source_code << tree_weights[ tree_mapping[i] ] << "f";
  }
  source_code << " };" << std::endl << std::endl;

  /*
  // print tree depths
  source_code << "const unsigned int tree_depths[N] = { ";
  for (size_t i = 0; i<tree_depths.size(); i++) {
    if (i!=0) source_code << ", ";
    source_code << tree_depths[ tree_mapping[i] ];
  }
  source_code << " };" << std::endl << std::endl;
   */

  // print leaf outputs
  source_code << "const double leaf_outputs[N][F] = { " << std::endl << '\t';
  for (size_t i = 0; i<tree_outputs.size(); i++) {
    if (i!=0) source_code << "," << std::endl << '\t';
    source_code << "\t{ ";
    for (size_t j = 0; j<tree_outputs[ tree_mapping[i] ].size(); j++) {
      if (j!=0) source_code << ", ";
      source_code << tree_outputs[ tree_mapping[i] ][j];
    }
    source_code << " }";
  }
  source_code << std::endl << "};" << std::endl << std::endl;

  // pint features ids
  source_code << "const unsigned int features_ids[N][M] = { " << std::endl << '\t';
  for (size_t i = 0; i<feature_ids.size(); i++) {
    if (i!=0) source_code << "," << std::endl << '\t';
    source_code << "\t{ ";
    for (size_t j = 0; j<feature_ids[ tree_mapping[i] ].size(); j++) {
      if (j!=0) source_code << ", ";
      source_code << feature_ids[ tree_mapping[i] ][j];
    }
    source_code << " }";
  }
  source_code << std::endl << "};" << std::endl << std::endl;


  // print thresholds values
  source_code << "const float thresholds[N][M] = { " << std::endl << '\t';
  source_code << std::setprecision(std::numeric_limits<Feature>::digits10);
  for (size_t i = 0; i<thresholds.size(); i++) {
    if (i!=0) source_code << "," << std::endl << '\t';
    source_code << "\t{ ";
    for (size_t j = 0; j<thresholds[ tree_mapping[i] ].size(); j++) {
      if (j!=0) source_code << ", ";
      source_code << thresholds[ tree_mapping[i] ][j] << "f";
    }
    source_code << " }";
  }
  source_code << std::endl << "};" << std::endl << std::endl;


  source_code << "#define SHL(n,p) ((n)<<(p))" << std::endl << std::endl;


  source_code
      << "unsigned int leaf_id(float *v, unsigned int const *fids, float const *thresh, const unsigned int m) {"
      << std::endl << "  unsigned int leafidx = 0;" << std::endl
      << "  for (unsigned int i=0; i<m; ++i)" << std::endl
      << "    leafidx |= SHL( v[fids[i]]>thresh[i], m-1-i);" << std::endl
      << "  return leafidx;" << std::endl << "}" << std::endl << std::endl;

  source_code << "double ranker(float *v) {" << std::endl
              << "  double score = 0.0;" << std::endl
              << "  int i = 0;" << std::endl;
  for (int d=0; d<max_depth; d++) {
    source_code << "  for (int j = 0; j < "<< depths_pupolation[d] <<"; ++j) {" << std::endl;
    source_code << "    score += tree_weights[i] * leaf_outputs[i][leaf_id(v, features_ids[i], thresholds[i], " << d+1 <<")];" << std::endl;
    source_code << "    i++;" << std::endl;
    source_code << "  }" << std::endl;
  }
  source_code << "  return score;" << std::endl << "}" << std::endl;

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
        new learning::forests::ObliviousLambdaMart(info_ptree, ensemble_ptree));

  return NULL;
}

}  // namespace io
}  // namespace quickrank
