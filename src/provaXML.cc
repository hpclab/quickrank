#include <string>
#include <memory>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <limits>
#include <list>
#include <iostream>
#include <cstring>
#include <utils/strutils.h>
#include <vector>
#include <numeric>
#include <algorithm>
#include <stdlib.h>

#include "types.h"
#include "pugixml/src/pugixml.hpp"

void generate_oblivious_code(const std::string, const std::string);
//void generate_conditional_operators_code(const std::string, const std::string);
//void model_node_to_conditional_operators(pugi::xml_node &, std::stringstream &os);

int main(int argc, char *argv[]) {
  std::string model_filename = "/Users/nardini/prova1.xml";
  std::string code_filename = "/Users/nardini/prova1.cc";
  generate_oblivious_code(model_filename, code_filename);
  return 1;
}

void model_tree_get_leaves(pugi::xml_node &split_xml,
                           std::vector<std::string> &leaves) {
  std::string prediction;
  bool is_leaf = false;
  pugi::xml_node left;
  pugi::xml_node right;

  for (pugi::xml_node &node : split_xml.children()) {
    if (strcmp(node.name(), "output") == 0) {
      prediction = node.child_value();
      trim(prediction);
      is_leaf = true;
      break;
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
    leaves.push_back(prediction);
  else {
    model_tree_get_leaves(left, leaves);
    model_tree_get_leaves(right, leaves);
  }
}

void generate_oblivious_code(const std::string model_filename,
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

  // let's navigate the ensemble, for each tree...
  pugi::xml_node ranker = xml_document.child("ranker");
  pugi::xml_node info = ranker.child("info");
  unsigned int trees = info.child("trees").text().as_uint();
  unsigned int depth = info.child("depth").text().as_uint();
  unsigned int max_leaves = 1 << depth;

  // printing ensemble info
  source_code << "#define N " << trees << " // no. of trees" << std::endl;
  source_code << "#define M " << depth << " // max tree depth" << std::endl;
  source_code << "#define F " << max_leaves << " // max number of leaves"
      << std::endl;
  source_code << std::endl;

  pugi::xml_node ensemble = ranker.child("ensemble");

  // loading tree weights
  std::vector<float> tree_weights;
  for (pugi::xml_node &tree : ensemble.children("tree")) {
    tree_weights.push_back(tree.attribute("weight").as_float());
  }

  // loading tree depths
  std::vector<int> tree_depths;
  for (pugi::xml_node &tree : ensemble.children("tree")) {
    int curr_depth = 0;
    auto splits_in_tree = tree.children("split");
    int size = std::distance(splits_in_tree.begin(), splits_in_tree.end());
    while (size != 2) {
      curr_depth++;
      tree = tree.child("split");
      splits_in_tree = tree.children("split");
      size = std::distance(splits_in_tree.begin(), splits_in_tree.end());
    }
    tree_depths.push_back(curr_depth);
  }

  // load leaf outputs
  std::vector<std::vector<std::string>> tree_outputs(trees);
  size_t curr_tree = 0;
  for (pugi::xml_node &tree : ensemble.children("tree")) {
    pugi::xml_node root = tree.child("split");
    model_tree_get_leaves(root, tree_outputs[curr_tree++]);
  }

  // load features ids
  std::vector<std::vector<size_t>> feature_ids(trees);
  for (pugi::xml_node &tree : ensemble.children("tree")) {
    for (pugi::xml_node &split : tree.children("split")) {
      int size = std::distance(split.begin(), split.end());
      while (size != 2) {
        split.print(std::cout);
        feature_ids[curr_tree].push_back(
            split.child("feature").text().as_int() - 1);
        split = split.child("split");
        size = std::distance(split.begin(), split.end());
        std::cout << size;
      }
    }
  }

  // load thresholds values
  std::vector<std::vector<std::string>> thresholds(trees);
  curr_tree = 0;
  for (pugi::xml_node &tree : ensemble.children("tree")) {
    for (pugi::xml_node &split : tree.children("split")) {
      int size = std::distance(split.begin(), split.end());
      while (size != 2) {
        std::string threshold = split.child("threshold").child_value();
        trim(threshold);
        thresholds[curr_tree].push_back(threshold);
        split = split.child("split");
      }
    }
  }
  curr_tree++;

  // mapping of trees in sorted order by depths
  std::vector<size_t> tree_mapping(trees);
  std::iota(tree_mapping.begin(), tree_mapping.end(), 0);
  std::sort(tree_mapping.begin(), tree_mapping.end(),
            [&tree_depths](int a, int b) {
              return tree_depths[a] < tree_depths[b];
            });

  // number of trees for each depth
  std::vector<size_t> depths_pupolation;
  int max_depth = tree_depths[tree_mapping.back()];

  int curr_depth = 1;
  size_t start_position = 0;
  for (size_t i = 0; i < tree_mapping.size(); i++) {
    while (tree_depths[tree_mapping[i]] > curr_depth) {
      depths_pupolation.push_back(i - start_position);
      curr_depth++;
      start_position = i;
    }
    if (curr_depth == max_depth)
      break;
  }
  depths_pupolation.push_back(tree_mapping.size() - start_position);

  // print tree weights
  source_code.setf(std::ios::floatfield, std::ios::fixed);
  source_code << "const float tree_weights[N] = { ";
  for (size_t i = 0; i < tree_weights.size(); i++) {
    if (i != 0)
      source_code << ", ";
    source_code << tree_weights[tree_mapping[i]] << "f";
  }
  source_code << " };" << std::endl << std::endl;

  // print leaf outputs
  source_code << "const double leaf_outputs[N][F] = { " << std::endl << '\t';
  for (size_t i = 0; i < tree_outputs.size(); i++) {
    if (i != 0)
      source_code << "," << std::endl << '\t';
    source_code << "\t{ ";
    for (size_t j = 0; j < tree_outputs[tree_mapping[i]].size(); j++) {
      if (j != 0)
        source_code << ", ";
      source_code << tree_outputs[tree_mapping[i]][j];
    }
    source_code << " }";
  }
  source_code << std::endl << "};" << std::endl << std::endl;

  // pint features ids
  source_code << "const unsigned int features_ids[N][M] = { " << std::endl
      << '\t';
  for (size_t i = 0; i < feature_ids.size(); i++) {
    if (i != 0)
      source_code << "," << std::endl << '\t';
    source_code << "\t{ ";
    for (size_t j = 0; j < feature_ids[tree_mapping[i]].size(); j++) {
      if (j != 0)
        source_code << ", ";
      source_code << feature_ids[tree_mapping[i]][j];
    }
    source_code << " }";
  }
  source_code << std::endl << "};" << std::endl << std::endl;

  // print thresholds values
  source_code << "const float thresholds[N][M] = { " << std::endl << '\t';
  source_code
      << std::setprecision(std::numeric_limits<quickrank::Feature>::digits10);
  for (size_t i = 0; i < thresholds.size(); i++) {
    if (i != 0)
      source_code << "," << std::endl << '\t';
    source_code << "\t{ ";
    for (size_t j = 0; j < thresholds[tree_mapping[i]].size(); j++) {
      if (j != 0)
        source_code << ", ";
      source_code << thresholds[tree_mapping[i]][j] << "f";
    }
    source_code << " }";
  }
  source_code << std::endl << "};" << std::endl << std::endl;

  source_code << "#define SHL(n,p) ((n)<<(p))" << std::endl << std::endl;

  source_code
      << "unsigned int leaf_id(float *v, unsigned int const *fids, float const *thresh, const unsigned int m) {"
      << std::endl << "  unsigned int leafidx;" << std::endl
      << "  for (unsigned int i=0; i<m; ++i)" << std::endl
      << "    leafidx |= SHL( v[fids[i]]>thresh[i], m-1-i);" << std::endl
      << "  return leafidx;" << std::endl << "}" << std::endl << std::endl;

  source_code << "double ranker(float *v) {" << std::endl
      << "  double score = 0.0;" << std::endl << "  int i = 0;"
      << std::endl;
  for (int d = 0; d < max_depth; d++) {
    source_code << "  for (int j = 0; j < " << depths_pupolation[d]
        << "; ++j) {" << std::endl;
    source_code
        << "    score += tree_weights[i] * leaf_outputs[i][leaf_id(v, features_ids[i], thresholds[i], "
        << d + 1 << ")];" << std::endl;
    source_code << "    i++;" << std::endl;
    source_code << "  }" << std::endl;
  }
  source_code << "  return score;" << std::endl << "}" << std::endl;

  std::ofstream output;
  output.open(code_filename, std::ofstream::out);
  output << source_code.str();
  output.close();
}


/*
void generate_conditional_operators_code(const std::string model_filename, const std::string code_filename) {
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
  for (pugi::xml_node tree : ensemble.children("tree")) {
    //int id = tree.attribute("id").as_int();
    float tree_weight = tree.attribute("weight").as_float();
    pugi::xml_node tree_content = tree.child("split");
    if (tree_content) {
      source_code << std::endl << "\t\t + " << std::setprecision(3) << tree_weight << "f * ";
      model_node_to_conditional_operators(tree_content, source_code);
    }
  }
  source_code << ";" << std::endl << "}" << std::endl;

  std::ofstream output;
  output.open(code_filename, std::ofstream::out);
  output << source_code.str();
  output.close();
}

void model_node_to_conditional_operators(pugi::xml_node& nodes, std::stringstream &os) {
  int feature_id = 0;
  std::string threshold;
  std::string prediction;
  bool is_leaf = false;
  pugi::xml_node left;
  pugi::xml_node right;

  for (pugi::xml_node& node : nodes.children()) {
    if (strcmp(node.name(), "output") == 0) {
      prediction = node.child_value();
      trim(prediction);
      os << prediction;
      is_leaf = true;
      break;
    } else if (strcmp(node.name(), "feature") == 0) {
      feature_id = atoi(node.child_value());
    } else if (strcmp(node.name(), "threshold") == 0) {
      threshold = node.child_value();
      trim(threshold);
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
}*/