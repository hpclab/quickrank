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
#include "pugixml/src/pugixml.hpp"

namespace quickrank {
namespace learning {

class LTR_Algorithm {

 public:
  LTR_Algorithm() {
  }

  /// Generates a LTR_Algorithm instance from a previously saved XML model.
  LTR_Algorithm(const pugi::xml_document &model);

  /// Avoid inefficient copy constructor
  LTR_Algorithm(const LTR_Algorithm &other) = delete;
  /// Avoid inefficient copy assignment
  LTR_Algorithm &operator=(const LTR_Algorithm &) = delete;

  virtual ~LTR_Algorithm() {
  }

  /// Returns the name of the ranker.
  virtual std::string name() const = 0;

  /// Executes the learning process.
  ///
  /// \param training_dataset The training dataset.
  /// \param validation_dataset The validation training dataset.
  /// \param metric The metric to be optimized.
  /// \param partial_save Allows to save a partial model every given number of iterations.
  /// \param model_filename The file where the model, and the partial models, are saved.
  virtual void learn(std::shared_ptr<data::Dataset> training_dataset,
                     std::shared_ptr<data::Dataset> validation_dataset,
                     std::shared_ptr<metric::ir::Metric> metric,
                     size_t partial_save,
                     const std::string model_filename) = 0;

  /// Given and input \a dateset, the current ranker generates
  /// scores for each instance and store the in the \a scores vector.
  ///
  /// \param dataset The dataset to be scored.
  /// \param scores The vector where scores are stored.
  /// \note Before scoring it invokes the function \a preprocess_dataset.
  ///       Usually this does not need to be overridden.
  virtual void score_dataset(std::shared_ptr<data::Dataset> dataset,
                             Score *scores) const;

  /// Returns the score of a given document.
  /// \param d is a pointer to the document to be evaluated
  /// \note   Each algorithm has a different implementation.
  virtual Score score_document(const Feature *d) const = 0;

  /// Returns the partial score of a given document, tree by tree.
  /// \param d is a pointer to the document to be evaluated
  /// \param next_fx_offset The offset to the next feature in the data representation.
  /// \note   Each algorithm has a different implementation.
  virtual std::shared_ptr<std::vector<Score>> partial_scores_document(
      const Feature *d, bool ignore_weights=false) const {
    return nullptr;
  }

  /// Save the current model to the output_file.
  ///
  /// \param model_filename The output file name.
  /// \param suffix The suffix used to identify partial model saves.
  virtual void save(std::string model_filename, int suffix = -1) const;

  /// Load a model from a given XML file.
  ///
  /// \param model_filename The input file name.
  static std::shared_ptr<LTR_Algorithm> load_model_from_file(
      std::string model_filename);

  /// Load a LtR model from a given XML model.
  ///
  /// \param xml_model The input file name.
  static std::shared_ptr<LTR_Algorithm> load_model_from_xml(
      const pugi::xml_document& xml_model);

  /// Import the state of the model from a previously trained model object
  /// (this operation overwrite the current state)
  /// Default implementation will do nothing (default for non ensemble models).
  /// WARNING: after the call, the passed LtR algo will be useless
  /// (inconsistent state).
  ///
  /// \param other The model from which to copy the model state
  /// \return bool indicating if the operation was succesfull
  virtual bool import_model_state(LTR_Algorithm &other) {
    return false;
  };

  /// Return the xml model representing the current object
  virtual pugi::xml_document *get_xml_model() const = 0;

  /// Print additional statistics.
  ///
  /// At the moment this include only number of comparisons for tree-based algorithms.
  virtual void print_additional_stats(void) const {
  }

  /// Update the weights for the ensemble models (only).
  ///
  /// Default implementation will do nothing (default for non ensemble models).
  virtual bool update_weights(std::vector<double>& weights) {
    return false;
  }

  /// Return the weights for the ensemble models (only).
  ///
  /// Default implementation will do nothing (default for non ensemble models).
  virtual std::vector<double> get_weights() const {
    // empty std::vector
    return {};
  }

 private:

  /// The output stream operator.
  friend std::ostream &operator<<(std::ostream &os, const LTR_Algorithm &a) {
    return a.put(os);
  }

  /// Prints the description of Algorithm, including its parameters
  virtual std::ostream &put(std::ostream &os) const = 0;

};

}  // namespace learning
}  // namespace quickrank
