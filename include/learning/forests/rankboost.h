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
 *   HPC. Laboratory - ISTI - CNR - http://hpc.isti.cnr.it/
 *   Tommaso Papini - https://github.com/oddlord
 *   Gabriele Bani - https://github.com/brnibani
 */
#pragma once

#include <memory>

#include "data/dataset.h"
#include "metric/ir/metric.h"
#include "learning/ltr_algorithm.h"

namespace quickrank {
namespace learning {
namespace forests {

class WeakRanker {
 public:

  WeakRanker(unsigned int feature_id, Feature theta, int sign) {
    feature_id_ = feature_id;
    theta_ = theta;
    sign_ = sign;
  }

  ~WeakRanker() {
  }

  unsigned int get_feature_id() const {
    return feature_id_;
  }

  Feature get_theta() const {
    return theta_;
  }

  int get_sign() const {
    return sign_;
  }

  unsigned int score_document(const quickrank::Feature *d) {
    if (sign_ * d[feature_id_] > sign_ * theta_)
      return 1;
    return 0;
  }

  WeakRanker *clone() {
    return new WeakRanker(feature_id_, theta_, sign_);
  }

 private:
  unsigned int feature_id_ = 0;
  Feature theta_ = 0.0;
  int sign_ = 1;

  friend std::ostream &operator<<(std::ostream &os, const WeakRanker &a) {
    return a.put(os);
  }

  std::ostream &put(std::ostream &os) const {
    os << "# WeakRanker " << feature_id_ << ":" << theta_ << " (" << ")"
       << std::endl;
    return os;
  }

};

/// This implements the RankBoost algorithm.
///
/// Freund, Y., Iyer, R., Schapire, R. E., & Singer, Y. (2003).
/// An efficient boosting algorithm for combining preferences.
/// The Journal of machine learning research, 4, 933-969.
class Rankboost: public LTR_Algorithm {
 public:
  Rankboost(size_t max_wr);

  Rankboost(const pugi::xml_document &model);

  virtual ~Rankboost();

  /// Avoid inefficient copy constructor
  Rankboost(const Rankboost &other) = delete;
  /// Avoid inefficient copy assignment
  Rankboost &operator=(const Rankboost &) = delete;

  /// Returns the name of the ranker.
  virtual std::string name() const {
    return NAME_;
  }

  static const std::string NAME_;

  /// Executes the learning process.
  ///
  /// \param training_dataset The training dataset.
  /// \param validation_dataset The validation training dataset.
  /// \param metric The metric to be optimized.
  /// \param partial_save Allows to save a partial model every given number of iterations.
  /// \param model_filename The file where the model, and the partial models, are saved.
  virtual void learn(
      std::shared_ptr<data::Dataset> training_dataset,
      std::shared_ptr<data::Dataset> validation_dataset,
      std::shared_ptr<metric::ir::Metric> metric,
      size_t partial_save,
      const std::string model_filename);

  /// Returns the score of a given document.
  virtual Score score_document(const Feature *d) const;

  /// Returns the partial scores of a given document, tree.
  /// \param d is a pointer to the document to be evaluated
  virtual std::shared_ptr<std::vector<Score>> partial_scores_document(
      const Feature *d, bool ignore_weights=false) const;

  /// Return the xml model representing the current object
  virtual pugi::xml_document *get_xml_model() const;

  virtual bool update_weights(std::vector<double> &weights);

  virtual std::vector<double> get_weights() const;

 private:
  float ***D = NULL;
  float **PI = NULL;
  Feature **THETA = NULL;
  unsigned int *n_theta = NULL;
  unsigned int ***SDF = NULL;
  Score *training_scores = NULL;
  Score *validation_scores = NULL;
  size_t T;
  size_t best_T;
  bool go_parallel;
  char const *omp_schedule;
  WeakRanker **weak_rankers = NULL;
  float *alphas = NULL;
  float best_r = 0.0;
  float max_alpha = 0.0;
  float r_t = 0.0;
  float z_t = 1.0;

  void init(std::shared_ptr<data::Dataset> training_dataset,
            std::shared_ptr<data::Dataset> validation_dataset);
  void compute_pi(std::shared_ptr<data::Dataset> dataset);
  WeakRanker *compute_weak_ranker(std::shared_ptr<data::Dataset> dataset);
  void update_d
      (std::shared_ptr<data::Dataset> dataset, WeakRanker *wr, float alpha);
  MetricScore compute_metric_score(std::shared_ptr<data::Dataset> dataset,
                                   std::shared_ptr<quickrank::metric::ir::Metric> scorer);
  void clean(std::shared_ptr<data::Dataset> dataset);


  /// The output stream operator.

  friend std::ostream &operator<<(std::ostream &os, const Rankboost &a) {
    return a.put(os);
  }

  /// Prints the description of Algorithm, including its parameters
  virtual std::ostream &put(std::ostream &os) const;
};
} // namespace forests
} // namespace learning
} // namespace quickrank

