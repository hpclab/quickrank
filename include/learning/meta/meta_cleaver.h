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

#include "types.h"
#include "learning/ltr_algorithm.h"
#include "learning/tree/rt.h"
#include "learning/tree/ensemble.h"
#include "optimization/optimization.h"
#include "optimization/post_learning/post_learning_opt.h"
#include "optimization/post_learning/cleaver/cleaver.h"


namespace quickrank {
namespace learning {
namespace meta {

class MetaCleaver: public LTR_Algorithm {
 public:
  /// Initializes a new Meta Cleaver instance with the given learning
  /// parameters.
  ///
  /// \param ltr_algo LtR algorithm to use for training.
  /// \param opt_algo Optimization algorithm to use for optimizing the model.
  /// \param ntrees Maximum number of trees.
  /// \param ntrees_per_iter Maximum number of trees to train for each iter.
  MetaCleaver(
      std::shared_ptr<learning::LTR_Algorithm> ltr_algo,
      std::shared_ptr<optimization::post_learning::pruning::Cleaver> cleaver,
      size_t ntrees,
      size_t ntrees_per_iter,
      double pruning_rate_per_iter,
      bool opt_last_only,
      size_t valid_iterations,
      bool verbose = false)
      : ltr_algo_(ltr_algo),
        cleaver_(cleaver),
        ntrees_(ntrees),
        ntrees_per_iter_(ntrees_per_iter),
        pruning_rate_per_iter_(pruning_rate_per_iter),
        opt_last_only_(opt_last_only),
        valid_iterations_(valid_iterations),
        verbose_(verbose) { }

  /// Generates a LTR_Algorithm instance from a previously saved XML model.
  MetaCleaver(const pugi::xml_document &model);

  virtual ~MetaCleaver() {
  };

  /// Start the learning process.
  virtual void learn(std::shared_ptr<data::Dataset> training_dataset,
                     std::shared_ptr<data::Dataset> validation_dataset,
                     std::shared_ptr<metric::ir::Metric> training_metric,
                     size_t partial_save,
                     const std::string output_basename);

  /// Returns the score by the current ranker
  ///
  /// \param d Document to be scored.
  virtual Score score_document(const Feature *d) const {
    return ltr_algo_->score_document(d);
  }

  /// Returns the partial scores of a given document, tree.
  /// \param d is a pointer to the document to be evaluated
  /// \param next_fx_offset The offset to the next feature in the data representation.
  /// \note   Each algorithm has a different implementation.
  virtual std::shared_ptr<std::vector<Score>> partial_scores_document(
      const Feature *d, bool ignore_weights=false) const {
    return ltr_algo_->partial_scores_document(d, ignore_weights);
  }

  /// Print additional statistics.
  ///
  /// Do Nothing right now
  virtual void print_additional_stats(void) const {
  };

  /// Returns the name of the ranker.
  virtual std::string name() const {
    return NAME_;
  }

  virtual bool update_weights(std::vector<double>& weights) {
    return ltr_algo_->update_weights(weights);
  };

  virtual std::vector<double> get_weights() const {
    return ltr_algo_->get_weights();
  }

  virtual bool import_model_state(LTR_Algorithm &other);

  static const std::string NAME_;

 protected:

  virtual pugi::xml_document *get_xml_model() const;

 protected:
  std::shared_ptr<quickrank::learning::LTR_Algorithm> ltr_algo_;
  std::shared_ptr<quickrank::optimization::post_learning::pruning::Cleaver>
      cleaver_;
  size_t ntrees_;
  size_t ntrees_per_iter_;
  double pruning_rate_per_iter_;
  bool opt_last_only_;
  size_t valid_iterations_; // If no performance gain on validation data is
                            // observed in 'esr' rounds, stop the training
                            // process right away (if esr==0 feature is disabled)
  bool verbose_;

 private:
  /// The output stream operator.
  friend std::ostream &operator<<(std::ostream &os, const MetaCleaver &a) {
    return a.put(os);
  }

  /// Prints the description of Algorithm, including its parameters.
  virtual std::ostream &put(std::ostream &os) const;

};

}  // namespace forests
}  // namespace learning
}  // namespace quickrank

