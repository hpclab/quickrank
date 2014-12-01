/*
 * QuickRank - A C++ suite of Learning to Rank algorithms
 * Webpage: http://quickrank.isti.cnr.it/
 * Contact: quickrank@isti.cnr.it
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
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

 private:

};

}  // namespace io
}  // namespace quickrank

#endif
