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
#ifndef QUICKRANK_LEARNING_LTR_ALGORITHM_H_
#define QUICKRANK_LEARNING_LTR_ALGORITHM_H_

#include <boost/noncopyable.hpp>
#include <boost/property_tree/ptree.hpp>
#include <memory>

#include "data/dataset.h"
#include "metric/ir/metric.h"

namespace quickrank {
namespace learning {

class LTR_Algorithm : private boost::noncopyable {

 public:
  LTR_Algorithm() {
  }

  /// Generates a LTR_Algorithm instance from a previously saved XML model.
  LTR_Algorithm(const boost::property_tree::ptree &info_ptree,
                const boost::property_tree::ptree &model_ptree);

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
                     unsigned int partial_save,
                     const std::string model_filename) = 0;

  /// Given and input \a dateset, the current ranker generates
  /// scores for each instance and store the in the \a scores vector.
  ///
  /// \param dataset The dataset to be scored.
  /// \param scores The vector where scores are stored.
  /// \note Before scoring it invokes the function \a preprocess_dataset.
  ///       Usually this does not need to be overridden.
  virtual void score_dataset(std::shared_ptr<data::Dataset> dataset,
                             Score* scores) const;

  /// Computes \a scores for a given set of documents.
  ///
  /// \param results The results list to be evaluated
  /// \param scores The vector where scores are stored.
  /// \param next_fx_offset The offset to the next feature in the data representation.
  /// \param next_d_offset The offset to the next document in the data representation.
  /// \note  Usually this does not need to be overridden.
  virtual void score_query_results(std::shared_ptr<data::QueryResults> results,
                                   Score* scores,
                                   unsigned int next_fx_offset,
                                   unsigned int next_d_offset) const;

  /// Returns the score of a given document.
  /// \param d is a pointer to the document to be evaluated
  /// \param next_fx_offset The offset to the next feature in the data representation.
  /// \note   Each algorithm has a different implementation.
  virtual Score score_document(const Feature* d,
                               const unsigned int next_fx_offset) const = 0;

  /// Returns the partial score of a given document, tree by tree.
  /// \param d is a pointer to the document to be evaluated
  /// \param next_fx_offset The offset to the next feature in the data representation.
  /// \note   Each algorithm has a different implementation.
  virtual std::shared_ptr<std::vector<Score>> detailed_scores_document(
      const Feature* d,
      const unsigned int next_fx_offset) const = 0;

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

  /// Save the current model in the given output file stream.
  virtual std::ofstream& save_model_to_file(std::ofstream& of) const = 0;

  /// Print additional statistics.
  ///
  /// At the moment this include only number of comparisons for tree-based algorithms.
  virtual void print_additional_stats(void) const {
  }

 protected:

  /// Prepare the dataset before training or scoring takes place.
  ///
  /// Different algorithms might modify the data representation
  /// to improve efficacy or efficiency,
  /// This is also used to make sure dataset is in the right vertical vs. horizontal format.
  virtual void preprocess_dataset(
      std::shared_ptr<data::Dataset> dataset) const = 0;

 private:

  /// The output stream operator.
  friend std::ostream& operator<<(std::ostream& os, const LTR_Algorithm& a) {
    return a.put(os);
  }

  /// Prints the description of Algorithm, including its parameters
  virtual std::ostream& put(std::ostream& os) const = 0;

};

}  // namespace learning
}  // namespace quickrank

#endif
