#ifndef QUICKRANK_LEARNING_CUSTOM_LTR_H_
#define QUICKRANK_LEARNING_CUSTOM_LTR_H_

#include <boost/noncopyable.hpp>
#include <memory>

#include "data/dataset.h"
#include "metric/ir/metric.h"
#include "learning/ltr_algorithm.h"

namespace quickrank {
namespace learning {

/*
 * Command lin
 ./bin/quickrank --algo custom \
 --train tests/data/msn1.fold1.train.5k.txt \
 --valid tests/data/msn1.fold1.vali.5k.txt \
 --test tests/data/msn1.fold1.test.5k.txt \
 --model model.xml
*/

class CustomLTR : public LTR_Algorithm {

 public:
  CustomLTR();

  virtual ~CustomLTR();

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
  virtual void score_query_results(
      std::shared_ptr<data::QueryResults> results,
      Score* scores, unsigned int offset) const;

  /// Returns the score of a given document.
  virtual Score score_document(const Feature* d,
                               const unsigned int offset = 1) const;


  /// \todo TODO: add load_model();


  const Score FIXED_SCORE = 666.0;

 protected:

  /// Prepare the dataset before training or scoring takes place.
  ///
  /// Different algorithms might modify the data representation
  /// to improve efficacy or efficiency,
  /// This is also used to make sure dataset is in the right vertical vs. horizontal format.
  virtual void preprocess_dataset(std::shared_ptr<data::Dataset> dataset) const;

 private:

  /// The output stream operator.
  friend std::ostream& operator<<(std::ostream& os, const CustomLTR& a) {
    return a.put(os);
  }

  /// Prints the description of Algorithm, including its parameters
  virtual std::ostream& put(std::ostream& os) const;

  /// Save the current model in the given output file stream.
  virtual std::ofstream& save_model_to_file(std::ofstream& of) const;
};

}  // namespace learning
}  // namespace quickrank

#endif
