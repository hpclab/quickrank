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

#include "learning/tree/rtnode.h"

#ifdef QUICKRANK_PERF_STATS
std::atomic<std::uint_fast64_t>RTNode::_internal_nodes_traversed = {0};
#endif

void RTNode::save_leaves(RTNode **&leaves, size_t &nleaves,
                         size_t &capacity) {
  if (featureidx == uint_max) {
    if (nleaves == capacity) {
      capacity = 2 * capacity + 1;
      leaves = (RTNode**) realloc(leaves, sizeof(RTNode*) * capacity);
    }
    leaves[nleaves++] = this;
  } else {
    left->save_leaves(leaves, nleaves, capacity);
    right->save_leaves(leaves, nleaves, capacity);
  }
}

std::shared_ptr<pugi::xml_node> RTNode::get_xml_model(
    const std::string& pos) const {

  std::stringstream ss;
  ss << std::setprecision(std::numeric_limits<double>::digits10);

  pugi::xml_node* split = new pugi::xml_node();
  split->set_name("split");

  if (!pos.empty())
    split->append_attribute("pos") = pos.c_str();

  if (featureid == uint_max) {

    ss << avglabel;
    split->append_child("output").text() = ss.str().c_str();

  } else {

    split->append_child("feature").text() = featureid;

    ss << threshold;
    split->append_child("threshold").text() = ss.str().c_str();

    split->append_move(*left->get_xml_model("left"));
    split->append_move(*right->get_xml_model("right"));
  }

  return std::shared_ptr<pugi::xml_node>(split);
}
