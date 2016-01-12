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
 * Contributors:
 *  - Andrea Battistini (andreabattistini@hotmail.com)
 *  - Chiara Pierucci (chiarapierucci14@gmail.com)
 *  - Claudio Lucchese (claudio.lucchese@isti.cnr.it)
 */
#ifndef QUICKRANK_LEARNING_ENSEMBLE_PRUNING_H_
#define QUICKRANK_LEARNING_ENSEMBLE_PRUNING_H_

#include <boost/noncopyable.hpp>
#include <boost/property_tree/ptree.hpp>
#include <memory>
#include <learning/linear/line_search.h>

#include "data/dataset.h"
#include "metric/ir/metric.h"
#include "learning/ltr_algorithm.h"

namespace quickrank {
namespace pruning {

/// This implements various strategies for pruning ensembles.
class EnsemblePruning : public quickrank::learning::LTR_Algorithm {

 public:

  enum class PruningMethod {
    RANDOM, LOW_WEIGHTS, SKIP, LAST, QUALITY_LOSS, AGGR_SCORE
  };

  EnsemblePruning(PruningMethod pruning_method, double pruning_rate);

  EnsemblePruning(std::string pruning_method, double pruning_rate);

  EnsemblePruning(std::string pruning_method, double pruning_rate,
                  std::shared_ptr<learning::linear::LineSearch> lineSearch);

  EnsemblePruning(const boost::property_tree::ptree &info_ptree,
                  const boost::property_tree::ptree &model_ptree);

  virtual ~EnsemblePruning();

  static PruningMethod getPruningMethod(std::string name) {
    auto i_item = std::find(pruningMethodName.cbegin(),
                            pruningMethodName.cend(),
                            name);
    if (i_item != pruningMethodName.cend()) {

      return PruningMethod(std::distance(pruningMethodName.cbegin(), i_item));
    }

    throw std::invalid_argument("pruning method name is not valid");
  }

  static std::string getPruningMethod(PruningMethod pruningMethod) {
    return pruningMethodName[static_cast<int>(pruningMethod)];
  }

  static const std::string NAME_;

  /// Returns the name of the ranker.
  virtual std::string name() const {
    return NAME_;
  }

  /// Returns the pruning method of the algorithm.
  virtual PruningMethod type() const {
    return pruning_method_;
  }

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
                     const std::string model_filename);

  /*
  /// Given and input \a dateset, the current ranker generates
  /// scores for each instance and store the in the \a scores vector.
  ///
  /// \param dataset The dataset to be scored.
  /// \param scores The vector where scores are stored.
  /// \note Before scoring it transposes the dataset in vertical format
  virtual void score_dataset(std::shared_ptr<data::Dataset> dataset,
                             Score* scores) const;

  /// Computes \a scores for a given set of documents.
  ///
  /// \param results The results list to be evaluated
  /// \param scores The vector where scores are stored.
  /// \param offset The offset to the next feature in the data representation.
  virtual void score_query_results(std::shared_ptr<data::QueryResults> results,
                                   Score* scores, unsigned int offset) const;
  */

  /// Returns the score of a given document.
  virtual Score score_document(const Feature* d,
                               const unsigned int next_fx_offset) const;

  virtual std::shared_ptr<std::vector<Score>> detailed_scores_document(
      const Feature* d, const unsigned int next_fx_offset) const {
    return nullptr;
  }

  /// Process the dataset filtering out features with 0-weight
  virtual std::shared_ptr<data::Dataset> filter_dataset(
      std::shared_ptr<data::Dataset> dataset) const;

 protected:

  /// Prepare the dataset before training or scoring takes place.
  ///
  /// Different algorithms might modify the data representation
  /// to improve efficacy or efficiency,
  /// This is also used to make sure dataset is in the right vertical vs. horizontal format.
  virtual void preprocess_dataset(std::shared_ptr<data::Dataset> dataset) const;

 private:
  double pruning_rate_;
  PruningMethod pruning_method_;
  unsigned int estimators_to_select_;
  std::shared_ptr<learning::linear::LineSearch> lineSearch_;

  std::vector<double> weights_;

  static const std::vector<std::string> pruningMethodName;

  /// The output stream operator.
  friend std::ostream& operator<<(std::ostream& os, const EnsemblePruning& a) {
    return a.put(os);
  }

  /// Prints the description of Algorithm, including its parameters
  virtual std::ostream& put(std::ostream& os) const;

  /// Save the current model in the given output file stream.
  virtual std::ofstream& save_model_to_file(std::ofstream& of) const;

  virtual void score(data::Dataset *dataset, Score *scores) const;

  /// The various pruning strategies
  virtual void random_pruning(std::shared_ptr<data::Dataset> dataset);
};

}  // namespace pruning
}  // namespace quickrank

#endif
