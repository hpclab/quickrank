#ifndef QUICKRANK_LEARNING_FORESTS_MART_H_
#define QUICKRANK_LEARNING_FORESTS_MART_H_

#include "types.h"
#include "learning/forests/lambdamart.h"
#include "learning/tree/rt.h"
#include "learning/tree/ensemble.h"

namespace quickrank {
namespace learning {
namespace forests {

class Mart : public LambdaMart {
 public:
  /// Initializes a new Mart instance with the given learning parameters.
  ///
  /// \param ntrees Maximum number of trees.
  /// \param shrinkage Learning rate.
  /// \param nthresholds Number of bins in discretization. 0 means no discretization.
  /// \param ntreeleaves Maximum number of leaves in each tree.
  /// \param minleafsupport Minimum number of instances in each leaf.
  /// \param esr Early stopping if no improvement after \esr iterations
  /// on the validation set.
  Mart(unsigned int ntrees, float shrinkage, unsigned int nthresholds,
             unsigned int ntreeleaves, unsigned int minleafsupport,
             unsigned int esr)
 : LambdaMart(ntrees,shrinkage,nthresholds,ntreeleaves,minleafsupport,esr) {
  }

  ~Mart() {
    //const unsigned int nfeatures = training_dataset ? training_set->get_nfeatures() : 0;
    //if (sortedsid)
    //for (unsigned int i = 0; i < training_dataset->num_features(); ++i)
    //delete[] sortedsid[i], free(thresholds[i]);
    //delete[] thresholds, delete[] thresholds_size, delete[] trainingmodelscores, delete[] pseudoresponses, delete[] sortedsid, delete[] cachedweights;
    //delete hist;
    //delete[] scores_on_validation;
    // perche' si mischiano free e delete?
  }

  /// Start the learning process.
  /*
  void learn(std::shared_ptr<data::Dataset> training_dataset,
             std::shared_ptr<data::Dataset> validation_dataset,
             std::shared_ptr<metric::ir::Metric> training_metric,
             unsigned int partial_save,
             const std::string output_basename);
   */

  /// Returns the score by the current ranker
  ///
  /// \param d Document to be scored.
  /// \param offset Offset to the next feature from \a d.
  /*
  virtual Score score_document(const Feature* d,
                               const unsigned int offset = 1) const {
    return ens.score_instance(d, offset);
  }
  */

 protected:
  /// Makes sure the dataset in in vertical format.
  //virtual void preprocess_dataset(std::shared_ptr<data::Dataset> dataset) const;

  /// Computes pseudo responses.
  ///
  /// \param training_dataset The training data.
  /// \param metric The metric to be optimized.
  virtual void compute_pseudoresponses(std::shared_ptr<data::Dataset> training_dataset,
                               metric::ir::Metric* metric);

  virtual float update_tree_prediction(RegressionTree* tree);

  /// Updates scores with the last learnt regression tree.
  ///
  /// \param dataset Dataset to be scored.
  /// \param scores Scores vector to be updated.
  /// \param tree Last regression tree leartn.
  /*void update_modelscores(std::shared_ptr<data::Dataset> dataset,
                          Score *scores,
                          RegressionTree* tree);
  */

 private:
  /// Prepares private data structurs befor training takes place.
  /*void init(std::shared_ptr<data::Dataset> training_dataset,
            std::shared_ptr<data::Dataset> validation_dataset);
  */

  /// The output stream operator.
  friend std::ostream& operator<<(std::ostream& os, const Mart& a) {
    return a.put(os);
  }

  /// Prints the description of Algorithm, including its parameters.
  virtual std::ostream& put(std::ostream& os) const;

  //virtual std::ofstream& save_model_to_file(std::ofstream& os) const;

};

}  // namespace forests
}  // namespace learning
}  // namespace quickrank

#endif
