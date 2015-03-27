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

#include "learning/tree/rtnode.h"

#ifdef QUICKRANK_PERF_STATS
std::atomic<std::uint_fast64_t>RTNode::_internal_nodes_traversed = {0};
#endif

void RTNode::save_leaves(RTNode **&leaves, unsigned int &nleaves,
                         unsigned int &capacity) {
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

// TODO TO BE REMOVED
void RTNode::write_outputtofile(FILE *f, const int indentsize) {
  char* indent = new char[indentsize + 1];  // char indent[indentsize+1];
  for (int i = 0; i < indentsize; indent[i++] = '\t')
    ;
  indent[indentsize] = '\0';
  if (featureid == uint_max)
    fprintf(f, "%s\t<output> %.15f </output>\n", indent, avglabel);
  else {
    fprintf(f, "%s\t<feature> %u </feature>\n", indent, featureid);
    fprintf(f, "%s\t<threshold> %.8f </threshold>\n", indent, threshold);
    fprintf(f, "%s\t<split pos=\"left\">\n", indent);
    left->write_outputtofile(f, indentsize + 1);
    fprintf(f, "%s\t</split>\n", indent);
    fprintf(f, "%s\t<split pos=\"right\">\n", indent);
    right->write_outputtofile(f, indentsize + 1);
    fprintf(f, "%s\t</split>\n", indent);
  }
  delete[] indent;
}

std::ofstream& RTNode::save_model_to_file(std::ofstream& os,
                                          const int indentsize) {
  std::string indent = "";
  for (int i = 0; i < indentsize; i++)
    indent += "\t";
  if (featureid == uint_max) {
    os << std::setprecision(std::numeric_limits<quickrank::Score>::digits10);
    os << indent << "\t<output> " << avglabel << " </output>" << std::endl;
  } else {
    os << indent << "\t<feature> " << featureid << " </feature>" << std::endl;
    os << std::setprecision(std::numeric_limits<quickrank::Feature>::digits10);
    os << indent << "\t<threshold> " << threshold << " </threshold>"
       << std::endl;
    os << indent << "\t<split pos=\"left\">" << std::endl;
    left->save_model_to_file(os, indentsize + 1);
    os << indent << "\t</split>" << std::endl;
    os << indent << "\t<split pos=\"right\">" << std::endl;
    right->save_model_to_file(os, indentsize + 1);
    os << indent << "\t</split>" << std::endl;
  }
  return os;
}
