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
#include <utils/strutils.h>

#include "pugixml/src/pugixml.hpp"

namespace quickrank {
namespace io {

/**
 * This class is a code generator on QuickRank XML files.
 *
 * The XML format is used for loading and storing ranking models.
 */
class GenOpCond {
 public:

  GenOpCond() {}
  ~GenOpCond() {}

  /// Generates the C++ implementation of the model scoring function.
  /// This applies to tree forests and generates a cascade of conditional operators.
  ///
  /// \param model_filename Previously saved xml ranker model.
  /// \param code_filename Output source code file name.
  void
  generate_conditional_operators_code(const std::string, const std::string);
};

}  // namespace io
}  // namespace quickrank