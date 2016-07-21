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
#include <string.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <queue>
#include <cmath>

#include "io/generate_vpred.h"
#include "pugixml/src/pugixml.hpp"
#include "utils/strutils.h"

namespace quickrank {
namespace io {

std::string trim_node_value(const pugi::xml_node &node,
                            const std::string &label) {
  std::string res = node.child_value(label.c_str());
  return trim(res);
}

bool is_leaf(const pugi::xml_node &node) {
  return !node.child("output").empty();
}

uint32_t find_depth(const pugi::xml_node &split) {
  uint32_t ld = 0, rd = 0;
  for (auto split_child : split.children()) {
    if (strcmp(split_child.name(), "output") == 0)
      return 1;
    else if (strcmp(split_child.name(), "split") == 0) {
      std::string pos = split_child.attribute("pos").value();
      if (pos == "left") {
        ld = 1 + find_depth(split_child);
      } else {
        rd = 1 + find_depth(split_child);
      }
    }
  }
  return std::max(ld, rd);
}

struct tree_node {
  const pugi::xml_node node;
  uint32_t id, pid;
  bool left;

  std::string feature, theta, leaf;

  tree_node(const pugi::xml_node &node, uint32_t id, uint32_t pid,
            bool left, const std::string &parent_f = "")
      : node(node),
        id(id),
        pid(pid),
        left(left) {
    if (is_leaf(node)) {
      this->feature = parent_f;
      this->theta = "";
      this->leaf = trim_node_value(node, "output");
    } else {
      this->feature = trim_node_value(node, "feature");
      this->theta = trim_node_value(node, "threshold");
      this->leaf = "";
    }
  }
};

void GenVpred::generate_vpred_input(const std::string &ensemble_file,
                                    const std::string &output_file) {

  if (ensemble_file.empty()) {
    std::cerr << "!!! Model filename is empty." << std::endl;
    exit(EXIT_FAILURE);
  }

  // parse XML
  pugi::xml_document model;
  pugi::xml_parse_result result = model.load_file(ensemble_file.c_str());
  if (!result) {
    std::cerr << "!!! Model filename is not parsed correctly." << std::endl;
    exit(EXIT_FAILURE);
  }

  std::ofstream output;
  output.open(output_file, std::ofstream::out);

  double learning_rate = model.select_node("/ranker/info/shrinkage").node()
      .text().as_double();

  auto trees = model.select_nodes("/ranker/ensemble/tree");
  size_t num_trees = trees.size();

  output << num_trees << std::endl;  // print number of trees in the ensemble

  // for each tree
  for (const auto &tree_it : trees) {

    auto tree = tree_it.node();

    auto split_node = tree.child("split");
    uint32_t depth = find_depth(split_node) - 1;

    output << (depth) << std::endl;  // print the tree depth

    uint32_t tree_size = std::pow(2, depth) - 1;
    uint32_t local_id = 0;

    std::queue<tree_node> node_queue;

    // breadth first visit
    // TODO: Remove object copies with references
    for (node_queue.push(tree_node(split_node, local_id++, -1, false));
         !node_queue.empty(); node_queue.pop()) {
      auto node = node_queue.front();
      if (is_leaf(node.node)) {
        if (node.id >= tree_size) {
          output << "leaf" << " " << node.id << " " << node.pid << " "
                 << node.left << " " << (learning_rate * std::stod(node.leaf))
                 << std::endl;
        } else {
          output << "node" << " " << node.id << " " << node.pid << " "
                 << (std::stoi(node.feature) - 1) << " " << node.left << " "
                 << (learning_rate * std::stod(node.leaf)) << std::endl;
        }
      } else {
        if (node.id == 0) {
          output << "root" << " " << node.id << " "
                 << (std::stoi(node.feature) - 1) << " " << node.theta
                 << std::endl;  // print the root info
        } else {
          output << "node" << " " << node.id << " " << node.pid << " "
                 << (std::stoi(node.feature) - 1) << " " << node.left << " "
                 << node.theta << std::endl;
        }
        for (const auto &split_child : node.node.children()) {

          if (strcmp(split_child.name(), "split") == 0) {
            std::string pos = split_child.attribute("pos").value();
            node_queue.push(
                tree_node(split_child, local_id++, node.id, (pos == "left"),
                          node.feature));
          }
        }
      }
    }

    output << "end" << std::endl;
  }
  output.close();
}

}  // namespace io
}  // namespace quickrank
