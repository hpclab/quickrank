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

 private:
  /// The output stream operator.
  friend std::ostream& operator<<(std::ostream& os, const LambdaMart& a) {
    return a.put(os);
  }

  /// Prints the description of Algorithm, including its parameters.
  virtual std::ostream& put(std::ostream& os) const;

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

 public:
  const unsigned int ntrees;  //>0
  const double shrinkage;  //>0.0f
  const unsigned int nthresholds;  //if nthresholds==0 then no. of thresholds is not limited
  const unsigned int ntreeleaves;  //>0
  const unsigned int minleafsupport;  //>0
  const unsigned int esr;  //If no performance gain on validation data is observed in 'esr' rounds, stop the training process right away (if esr==0 feature is disabled).

 public:
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

  void learn(std::shared_ptr<quickrank::data::Dataset> training_dataset,
             std::shared_ptr<quickrank::data::Dataset> validation_dataset,
             quickrank::metric::ir::Metric*, unsigned int partial_save,
             const std::string output_basename);


  // assumes vertical dataset
  virtual quickrank::Score score_document(const Feature* d,
                                          const unsigned int offset = 1) const {
    return ens.score_instance(d, offset);
  }

 protected:
  float compute_modelscores(LTR_VerticalDataset const *samples, double *mscores,
                            RegressionTree const &tree, quickrank::metric::ir::Metric* scorer);

  void update_modelscores(quickrank::data::Dataset* dataset,
                          quickrank::Score *scores, RegressionTree* tree);

  std::unique_ptr<quickrank::Jacobian> compute_mchange(
      const ResultList &orig, const unsigned int offset, quickrank::metric::ir::Metric* scorer);

  // Changes by Cla:
  // - added processing of ranked list in ranked order
  // - added cut-off in measure changes matrix
  void compute_pseudoresponses(std::shared_ptr<quickrank::data::Dataset> training_dataset, quickrank::metric::ir::Metric* scorer);

 private:
  void init(std::shared_ptr<quickrank::data::Dataset> training_dataset,
            std::shared_ptr<quickrank::data::Dataset> validation_dataset);
  virtual std::ofstream& save_model_to_file(std::ofstream& os) const;
};

}  // namespace forests
}  // namespace learning
}  // namespace quickrank

#endif
