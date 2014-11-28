#include <boost/property_tree/xml_parser.hpp>
#include <boost/foreach.hpp>

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





void model_node_to_c_code(const boost::property_tree::ptree &split_xml,
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
    os << "( v[" << feature_id-1 << "] <= " << std::setprecision(15) << threshold;
    os << " ? ";
    model_node_to_c_code(*left, os);
    os << " : ";
    model_node_to_c_code(*right, os);
    os << " )";
  }
}


void Xml::generate_c_code_baseline(std::string model_filename, std::string code_filename) {
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
    float tree_weight = tree.second.get("<xmlattr>.weight",1.0f);

    // find the root of the tree
    boost::property_tree::ptree root;
    BOOST_FOREACH(const boost::property_tree::ptree::value_type& node, tree.second ) {
      if (node.first == "split") {
        source_code << std::endl << "\t\t + " << std::setprecision(3) << tree_weight << " * ";
        model_node_to_c_code(node.second, source_code);
      }
    }
  }
  source_code << ";" << std::endl << "}" << std::endl;

  std::ofstream output;
  output.open(code_filename, std::ofstream::out);
  output << source_code.str();
  output.close();
}


void Xml::generate_c_code_oblivious_trees(std::string model_filename, std::string code_filename) {
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

  // Forests info
//  #define N 1 //no. of trees
//  #define M 5 //max tree depth

  // Tree Weights
//float ws[N] = { 0.50000000 };

  std::cout << source_code.str();

  /*
  std::ofstream output;
  output.open(code_filename, std::ofstream::out);
  output << source_code.str();
  output.close();
  */
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
