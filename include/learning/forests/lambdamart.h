#ifndef QUICKRANK_LEARNING_FORESTS_LMART_H_
#define QUICKRANK_LEARNING_FORESTS_LMART_H_

#include "types.h"
#include "learning/ltr_algorithm.h"
#include "learning/tree/rt.h"
#include "learning/tree/ensemble.h"

namespace quickrank {
namespace learning {
namespace forests {

class LambdaMart : public LTR_Algorithm {
 public:
  /// Initializes a new LambdaMart instance with the given learning parameters.
  ///
  /// \param ntrees Maximum number of trees.
  /// \param shrinkage Learning rate.
  /// \param nthresholds Number of bins in discretization. 0 means no discretization.
  /// \param ntreeleaves Maximum number of leaves in each tree.
  /// \param minleafsupport Minimum number of instances in each leaf.
  /// \param esr Early stopping if no improvement after \esr iterations
  /// on the validation set.
  LambdaMart(unsigned int ntrees, float shrinkage, unsigned int nthresholds,
             unsigned int ntreeleaves, unsigned int minleafsupport,
             unsigned int esr)
 : ntrees(ntrees),
   shrinkage(shrinkage),
   nthresholds(nthresholds),
   ntreeleaves(ntreeleaves),
   minleafsupport(minleafsupport),
   esr(esr) {
  }

  ~LambdaMart() {
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
  void learn(std::shared_ptr<data::Dataset> training_dataset,
             std::shared_ptr<data::Dataset> validation_dataset,
             std::shared_ptr<metric::ir::Metric> training_metric,
             unsigned int partial_save,
             const std::string output_basename);


  /// Returns the score by the current ranker
  ///
  /// \param d Document to be scored.
  /// \param offset Offset to the next feature from \a d.
  virtual Score score_document(const Feature* d,
                               const unsigned int offset = 1) const {
    return ens.score_instance(d, offset);
  }

 protected:
  /// Makes sure the dataset in in vertical format.
  virtual void preprocess_dataset(std::shared_ptr<data::Dataset> dataset) const;

  /// Computes pseudo responses.
  ///
  /// \param training_dataset The training data.
  /// \param metric The metric to be optimized.
  void compute_pseudoresponses(std::shared_ptr<data::Dataset> training_dataset,
                               metric::ir::Metric* metric);

  /// Updates scores with the last learnt regression tree.
  ///
  /// \param dataset Dataset to be scored.
  /// \param scores Scores vector to be updated.
  /// \param tree Last regression tree leartn.
  void update_modelscores(std::shared_ptr<data::Dataset> dataset,
                          Score *scores,
                          RegressionTree* tree);

 private:
  /// Prepares private data structurs befor training takes place.
  void init(std::shared_ptr<data::Dataset> training_dataset,
            std::shared_ptr<data::Dataset> validation_dataset);

  /// The output stream operator.
  friend std::ostream& operator<<(std::ostream& os, const LambdaMart& a) {
    return a.put(os);
  }

  /// Prints the description of Algorithm, including its parameters.
  virtual std::ostream& put(std::ostream& os) const;

  virtual std::ofstream& save_model_to_file(std::ofstream& os) const;

 protected:
  float **thresholds = NULL;
  unsigned int *thresholds_size = NULL;
  double *trainingmodelscores = NULL;  //[0..nentries-1]
  unsigned int validation_bestmodel = 0;
  double *pseudoresponses = NULL;  //[0..nentries-1]
  double *cachedweights = NULL;  //corresponds to datapoint.cache
  unsigned int **sortedsid = NULL;
  unsigned int sortedsize = 0;
  RTRootHistogram *hist = NULL;
  Ensemble ens;

  quickrank::Score* scores_on_validation = NULL;  //[0..nentries-1]

  const unsigned int ntrees;  //>0
  const double shrinkage;  //>0.0f
  const unsigned int nthresholds;  //if nthresholds==0 then no. of thresholds is not limited
  const unsigned int ntreeleaves;  //>0
  const unsigned int minleafsupport;  //>0
  const unsigned int esr;  //If no performance gain on validation data is observed in 'esr' rounds, stop the training process right away (if esr==0 feature is disabled).



};

}  // namespace forests
}  // namespace learning
}  // namespace quickrank

#endif
