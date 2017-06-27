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
#include <iomanip>
#include <limits>
#include <sstream>
#include <cstring>

#include "learning/tree/rtnode.h"

#ifdef QUICKRANK_PERF_STATS
std::atomic<std::uint_fast64_t>RTNode::_internal_nodes_traversed = {0};
#endif

void RTNode::save_leaves(RTNode **&leaves, size_t &nleaves,
                         size_t &capacity) {
  if (featureidx == uint_max) {
    if (nleaves == capacity) {
      capacity = 2 * capacity + 1;
      leaves = (RTNode **) realloc(leaves, sizeof(RTNode *) * capacity);
    }
    leaves[nleaves++] = this;
  } else {
    left->save_leaves(leaves, nleaves, capacity);
    right->save_leaves(leaves, nleaves, capacity);
  }
}

pugi::xml_node RTNode::append_xml_model(pugi::xml_node parent,
                                        const std::string &pos) const {

  std::stringstream ss;

  pugi::xml_node split = parent.append_child("split");

  if (!pos.empty())
    split.append_attribute("pos") = pos.c_str();

  if (featureid == uint_max) {

    ss << std::setprecision(std::numeric_limits<double>::max_digits10);
    ss << avglabel;
    split.append_child("output").text() = ss.str().c_str();

  } else {

    split.append_child("feature").text() = featureid;

    ss << std::setprecision(std::numeric_limits<float>::max_digits10);
    ss << threshold;
    split.append_child("threshold").text() = ss.str().c_str();

    left->append_xml_model(split, "left");
    right->append_xml_model(split, "right");
  }

  return split;
}

RTNode *RTNode::parse_xml(const pugi::xml_node &split_xml) {
  RTNode *model_node = NULL;
  RTNode *left_child = NULL;
  RTNode *right_child = NULL;

  bool is_leaf = false;

  unsigned int feature_id = 0;
  quickrank::Feature threshold = 0.0f;
  quickrank::Score prediction = 0.0;

  for (const pugi::xml_node &split_child: split_xml.children()) {

    if (strcmp(split_child.name(), "output") == 0) {
      prediction = split_child.text().as_double();
      is_leaf = true;
      break;
    } else if (strcmp(split_child.name(), "feature") == 0) {
      feature_id = split_child.text().as_uint();
    } else if (strcmp(split_child.name(), "threshold") == 0) {
      threshold = split_child.text().as_float();
    } else if (strcmp(split_child.name(), "split") == 0) {
      std::string pos = split_child.attribute("pos").value();
      if (pos == "left")
        left_child = RTNode::parse_xml(split_child);
      else
        right_child = RTNode::parse_xml(split_child);
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
