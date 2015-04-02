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
#ifndef QUICKRANK_LEARNING_COORDINATE_ASCENT_H_
#define QUICKRANK_LEARNING_COORDINATE_ASCENT_H_

#include <boost/noncopyable.hpp>
#include <boost/property_tree/ptree.hpp>
#include <memory>

#include "data/dataset.h"
#include "metric/ir/metric.h"
#include "learning/ltr_algorithm.h"

namespace quickrank {
namespace learning {
namespace linear {

/// This implements the Coordinate Ascent algorithm.
///
/// Metzler, D., Croft, W.B.: Linear feature-based models for information retrieval.
/// Information Retrieval 10(3), 257–274 (2007)
class CoordinateAscent : public LTR_Algorithm {

 public:
  CoordinateAscent(unsigned int num_points, double window_size,
                   double reduction_factor, unsigned int max_iterations,
                   unsigned int max_failed_vali);

  CoordinateAscent(const boost::property_tree::ptree &info_ptree,
                   const boost::property_tree::ptree &model_ptree);

  virtual ~CoordinateAscent();

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
  virtual void learn(std::shared_ptr<data::Dataset> training_dataset,
                     std::shared_ptr<data::Dataset> validation_dataset,
                     std::shared_ptr<metric::ir::Metric> metric,
                     unsigned int partial_save,
                     const std::string model_filename);

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

  /// Returns the score of a given document.
  virtual Score score_document(const Feature* d,
                               const unsigned int offset = 1) const;

 protected:

  /// Prepare the dataset before training or scoring takes place.
  ///
  /// Different algorithms might modify the data representation
  /// to improve efficacy or efficiency,
  /// This is also used to make sure dataset is in the right vertical vs. horizontal format.
  virtual void preprocess_dataset(std::shared_ptr<data::Dataset> dataset) const;

 private:
  std::vector<double> best_weights_;

  unsigned int num_samples_;
  double window_size_;
  double reduction_factor_;
  unsigned int max_iterations_;
  unsigned int max_failed_vali_;

  /// The output stream operator.
  friend std::ostream& operator<<(std::ostream& os, const CoordinateAscent& a) {
    return a.put(os);
  }

  /// Prints the description of Algorithm, including its parameters
  virtual std::ostream& put(std::ostream& os) const;

  /// Save the current model in the given output file stream.
  virtual std::ofstream& save_model_to_file(std::ofstream& of) const;
};

}  // namespace linear
}  // namespace learning
}  // namespace quickrank

#endif
