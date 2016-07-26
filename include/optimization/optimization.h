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

#include <memory>

#include "data/dataset.h"
#include "metric/ir/metric.h"
#include "learning/ltr_algorithm.h"

namespace quickrank {
namespace optimization {

class Optimization {

 public:

  enum class OptimizationAlgorithm {
    CLEAVER
  };

  Optimization() {};

  /// Generates a LTR_Algorithm instance from a previously saved XML model.
  Optimization(const pugi::xml_document &model);

  virtual ~Optimization() = default;

  /// Avoid inefficient copy constructor
  Optimization(const Optimization &other) = delete;
  /// Avoid inefficient copy assignment
  Optimization &operator=(const Optimization &) = delete;

  /// Returns the name of the optimizer.
  virtual std::string name() const = 0;

  virtual bool need_partial_score_dataset() const = 0;

  /// Executes the optimization process.
  ///
  /// \param training_dataset The training dataset.
  /// \param validation_dataset The validation dataset.
  /// \param metric The metric to be optimized.
  /// \param partial_save Allows to save a partial model every given number of iterations.
  /// \param model_filename The file where the model, and the partial models, are saved.
  virtual void optimize(std::shared_ptr<learning::LTR_Algorithm> algo,
                        std::shared_ptr<data::Dataset> training_dataset,
                        std::shared_ptr<data::Dataset> validation_dataset,
                        std::shared_ptr<metric::ir::Metric> metric,
                        size_t partial_save,
                        const std::string model_filename) = 0;

  /// Save the current model to the output_file.
  ///
  /// \param model_filename The output file name.
  /// \param suffix The suffix used to identify partial model saves.
  virtual void save(std::string model_filename, int suffix = -1) const;

  /// Return the xml model representing the current object
  virtual pugi::xml_document *get_xml_model() const = 0;

  virtual bool is_pre_learning() const = 0;

  /// Load a model from a given XML file.
  ///
  /// \param model_filename The input file name.
  static std::shared_ptr<Optimization> load_model_from_file(
      std::string model_filename);

  // Static methods
  static const std::vector<std::string> optimizationAlgorithmNames;

  static OptimizationAlgorithm getOptimizationAlgorithm(std::string name) {
    auto i_item = std::find(optimizationAlgorithmNames.cbegin(),
                            optimizationAlgorithmNames.cend(),
                            name);
    if (i_item != optimizationAlgorithmNames.cend()) {

      return OptimizationAlgorithm(std::distance(optimizationAlgorithmNames.cbegin(),
                                                 i_item));
    }

    throw std::invalid_argument("pruning method " + name + " is not valid");
  }

  static std::string getPruningMethod(OptimizationAlgorithm optAlgo) {
    return optimizationAlgorithmNames[static_cast<int>(optAlgo)];
  }

 protected:

  /// The output stream operator.
  friend std::ostream &operator<<(std::ostream &os, const Optimization &a) {
    return a.put(os);
  }

  /// Prints the description of Algorithm, including its parameters
  virtual std::ostream &put(std::ostream &os) const = 0;

};

}  // namespace optimization
}  // namespace quickrank