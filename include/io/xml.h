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
#ifndef QUICKRANK_IO_XML_H_
#define QUICKRANK_IO_XML_H_

#include <boost/property_tree/ptree.hpp>
#include <memory>
#include <string>

#include "learning/ltr_algorithm.h"
#include "learning/tree/rt.h"

namespace quickrank {
namespace io {

RTNode* RTNode_parse_xml(const boost::property_tree::ptree &split_xml);

/**
 * This class implements IO on Xml files.
 *
 * The XML format is used for loading and storing ranking models
 */
class Xml : private boost::noncopyable {
 public:
  /// Creates a new Svml IO reader/writer.
  ///
  /// \param k The cut-off threshold.
  Xml() {
  }
  ~Xml() {
  }

  /// Generates the C++ implementation of the model scoring function.
  /// This applies to tree forests and generates a cascade of conditional operators.
  ///
  /// \param model_filename Previously saved xml ranker model.
  /// \param code_filename Output source code file name.
  void generate_c_code_baseline(std::string model_filename,
                                std::string code_filename);

  /// Generates the C++ implementation of the model scoring function.
  /// This applies to forests of oblivious trees and generates a smart level-wise evaluator.
  ///
  /// \param model_filename Previously saved xml ranker model.
  /// \param code_filename Output source code file name.
  void generate_c_code_oblivious_trees(std::string model_filename,
                                       std::string code_filename);

  /// Loads a LTR algorithm from a previously saved XML file
  ///
  /// \param model_filename The file name of the xml model.
  /// \returns A new instance of a \a LTR_Algorithm.
  std::shared_ptr<learning::LTR_Algorithm> load_model_from_file(
      std::string model_filename);

  /// Generates the C++ implementation of the model scoring function.
  /// This is en experimental method.
  ///
  /// \param model_filename Previously saved xml ranker model.
  /// \param code_filename Output source code file name.
  void generate_c_code_vectorized(std::string model_filename,
                                  std::string code_filename);
 private:

};

}  // namespace io
}  // namespace quickrank

#endif
