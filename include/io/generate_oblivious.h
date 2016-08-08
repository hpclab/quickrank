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
#pragma once

#include <string>
#include <memory>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <limits>
#include <list>
#include <iostream>
#include <cstring>
#include <vector>
#include <utils/strutils.h>

#include "pugixml/src/pugixml.hpp"

namespace quickrank {
namespace io {

/**
 * This class is a code generator on QuickRank XML files.
 *
 * The XML format is used for loading and storing ranking models.
 */
class GenOblivious {
 public:

  /// Generates the C++ implementation of the model scoring function.
  /// This applies to forests of oblivious trees.
  ///
  /// \param model_filename Previously saved XML ranker model.
  /// \param code_filename Output source code file name.
  void generate_oblivious_code(const std::string, const std::string);

 private:
  void model_tree_get_leaves(pugi::xml_node &, std::vector<std::string> &);
  void
  model_tree_get_feature_ids(pugi::xml_node &, std::vector<unsigned int> &);
  void model_tree_get_thresholds(pugi::xml_node &, std::vector<std::string> &);
};

}  // namespace io
}  // namespace quickrank