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

#include <vector>
#include <numeric>
#include <algorithm>
#include <types.h>
#include "io/generate_oblivious.h"

namespace quickrank {
namespace io {

void GenOblivious::model_tree_get_leaves(pugi::xml_node &split_xml,
                                         std::vector<std::string> &leaves) {
  std::string prediction;
  bool is_leaf = false;
  pugi::xml_node left;
  pugi::xml_node right;

  for (pugi::xml_node &node : split_xml.children()) {
    if (strcmp(node.name(), "output") == 0) {
      prediction = node.text().as_string();
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

void GenOblivious::model_tree_get_feature_ids(pugi::xml_node &split_xml,
                                              std::vector<unsigned int> &features) {
  unsigned int feature_id;
  bool is_leaf = false;
  pugi::xml_node left;
  //pugi::xml_node right;

  for (pugi::xml_node &node : split_xml.children()) {
    if (strcmp(node.name(), "feature") == 0) {
      feature_id = node.text().as_uint();
      // decreasing by 1 to remap features to zero
      features.push_back(feature_id - 1);
    } else if (strcmp(node.name(), "output") == 0) {
      is_leaf = true;
      break;
    } else if (strcmp(node.name(), "split") == 0) {
      std::string pos = node.attribute("pos").as_string();

      if (pos == "left") {
        left = node;
      }
      if (pos == "right") {
        // not needed, simmetric trees...
        //right = node;
      }
    }
  }

  if (is_leaf)
    return;
  else {
    model_tree_get_feature_ids(left, features);
    //model_tree_get_feature_ids(right, features);
  }
}

void GenOblivious::model_tree_get_thresholds(pugi::xml_node &split_xml,
                                             std::vector<std::string> &thresholds) {
  std::string threshold;
  bool is_leaf = false;
  pugi::xml_node left;
  //pugi::xml_node right;

  for (pugi::xml_node &node : split_xml.children()) {
    if (strcmp(node.name(), "threshold") == 0) {
      threshold = node.text().as_string();
      trim(threshold);
      thresholds.push_back(threshold);
    } else if (strcmp(node.name(), "output") == 0) {
      is_leaf = true;
      break;
    } else if (strcmp(node.name(), "split") == 0) {
      std::string pos = node.attribute("pos").as_string();

      if (pos == "left") {
        left = node;
      }
      if (pos == "right") {
        // not needed, simmetric trees...
        //right = node;
      }
    }
  }

  if (is_leaf)
    return;
  else {
    model_tree_get_thresholds(left, thresholds);
    //model_tree_get_thresholds(right, thresholds);
  }
}

void GenOblivious::generate_oblivious_code(const std::string model_filename,
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
  // trees below not used as it can report a bigger number of trees
  // in case we are using intermediate XML model dump files...
  //unsigned int trees = info.child("trees").text().as_uint();
  unsigned int depth = info.child("depth").text().as_uint();
  unsigned int max_leaves = 1 << depth;

  pugi::xml_node ensemble = ranker.child("ensemble");

  // loading tree weights
  std::vector<float> tree_weights;
  for (pugi::xml_node &tree : ensemble.children("tree")) {
    tree_weights.push_back(tree.attribute("weight").as_float());
  }

  // loading tree depths
  std::vector<int> tree_depths;
  for (pugi::xml_node tree : ensemble.children("tree")) {
    int curr_depth = 0;
    auto splits_in_tree = tree.children("split");
    int size = std::distance(splits_in_tree.begin(), splits_in_tree.end());
    while (size != 0) {
      curr_depth++;
      tree = tree.child("split");
      splits_in_tree = tree.child("split").children("split");
      size = std::distance(splits_in_tree.begin(), splits_in_tree.end());
    }
    tree_depths.push_back(curr_depth);
  }

  int actual_model_size = tree_depths.size();

  // load leaf outputs
  std::vector<std::vector<std::string>> tree_outputs(actual_model_size);
  unsigned int curr_tree = 0;
  for (pugi::xml_node tree : ensemble.children("tree")) {
    pugi::xml_node root = tree.child("split");
    model_tree_get_leaves(root, tree_outputs[curr_tree++]);
  }

  // load features ids
  curr_tree = 0;
  std::vector<std::vector<unsigned int>> feature_ids(actual_model_size);
  for (pugi::xml_node tree : ensemble.children("tree")) {
    pugi::xml_node root = tree.child("split");
    model_tree_get_feature_ids(root, feature_ids[curr_tree++]);
  }

  // load thresholds values
  curr_tree = 0;
  std::vector<std::vector<std::string>> thresholds(actual_model_size);
  for (pugi::xml_node tree : ensemble.children("tree")) {
    pugi::xml_node root = tree.child("split");
    model_tree_get_thresholds(root, thresholds[curr_tree++]);
  }

  // mapping of trees in sorted order by depths
  std::vector<size_t> tree_mapping(tree_depths.size());
  std::iota(tree_mapping.begin(), tree_mapping.end(), 0);
  std::sort(tree_mapping.begin(), tree_mapping.end(),
            [&tree_depths](int a, int b) {
              return tree_depths[a] < tree_depths[b];
            });

  // number of trees for each depth
  std::vector<size_t> depths_population;
  int max_depth = tree_depths[tree_mapping.back()];

  int curr_depth = 1;
  size_t start_position = 0;
  for (size_t i = 0; i < tree_mapping.size(); i++) {
    while (tree_depths[tree_mapping[i]] > curr_depth) {
      depths_population.push_back(i - start_position);
      curr_depth++;
      start_position = i;
    }
    if (curr_depth == max_depth)
      break;
  }
  depths_population.push_back(tree_mapping.size() - start_position);

  // start printing code to output stream...
  // printing ensemble info
  source_code << "#define N " << actual_model_size << " // no. of trees"
              << std::endl;
  source_code << "#define M " << depth << " // max tree depth" << std::endl;
  source_code << "#define F " << max_leaves << " // max number of leaves"
              << std::endl;
  source_code << std::endl;

  // print tree weights
  source_code << std::setprecision(std::numeric_limits<float>::max_digits10);
  source_code << "const float tree_weights[N] = { ";
  for (size_t i = 0; i < tree_weights.size(); i++) {
    if (i != 0)
      source_code << ", ";
    source_code << tree_weights[tree_mapping[i]];
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
      << std::setprecision(
          std::numeric_limits<quickrank::Feature>::max_digits10);
  for (size_t i = 0; i < thresholds.size(); i++) {
    if (i != 0)
      source_code << "," << std::endl << '\t';
    source_code << "\t{ ";
    for (size_t j = 0; j < thresholds[tree_mapping[i]].size(); j++) {
      if (j != 0)
        source_code << ", ";
      source_code << thresholds[tree_mapping[i]][j];
    }
    source_code << " }";
  }
  source_code << std::endl << "};" << std::endl << std::endl;

  source_code << "#define SHL(n,p) ((n)<<(p))" << std::endl << std::endl;

  source_code
      << "unsigned int leaf_id(float *v, unsigned int const *fids, float const *thresh, const unsigned int m) {"
      << std::endl << "  unsigned int leafidx=0;" << std::endl
      << "  for (unsigned int i=0; i<m; ++i)" << std::endl
      << "    leafidx |= SHL( v[fids[i]]>thresh[i], m-1-i);" << std::endl
      << "  return leafidx;" << std::endl << "}" << std::endl << std::endl;

  source_code << "double ranker(float *v) {" << std::endl
              << "  double score = 0.0;" << std::endl << "  int i = 0;"
              << std::endl;
  for (int d = 0; d < max_depth; d++) {
    source_code << "  for (int j = 0; j < " << depths_population[d]
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

}  // namespace io
}  // namespace quickrank