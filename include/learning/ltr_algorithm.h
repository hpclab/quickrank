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
  LTR_Algorithm() {};

  /// Generates a LTR_Algorithm instance from a previously saved XML model.
  LTR_Algorithm(const boost::property_tree::ptree &info_ptree, const boost::property_tree::ptree &model_ptree);

  virtual ~LTR_Algorithm() {}


  /// Returns the name of the ranker.
  virtual std::string name() const = 0;

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
      const std::string model_filename) = 0;


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

  /// Save the current model to the output_file.
  ///
  /// \param model_filename The output file name.
  /// \param suffix The suffix used to identify partial model saves.
  virtual void save(std::string model_filename, int suffix = -1) const;

  /// Load a model from a given XML file.
  ///
  /// \param model_filename The input file name.
  static LTR_Algorithm* load_model_from_file(std::string model_filename);

  /// Save the current model in the given output file stream.
  virtual std::ofstream& save_model_to_file(std::ofstream& of) const = 0;

 protected:

  /// Prepare the dataset before training or scoring takes place.
  ///
  /// Different algorithms might modify the data representation
  /// to improve efficacy or efficiency,
  /// This is also used to make sure dataset is in the right vertical vs. horizontal format.
  virtual void preprocess_dataset(std::shared_ptr<data::Dataset> dataset) const = 0;

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
