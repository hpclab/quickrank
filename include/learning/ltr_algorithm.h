#ifndef QUICKRANK_LEARNING_RANKER_H_
#define QUICKRANK_LEARNING_RANKER_H_

#include "learning/dpset.h"
#include "utils/qsort.h"

#include "metric/ir/metric.h"

namespace quickrank {
namespace learning {

class LTR_Algorithm {



 protected:
  qr::metric::ir::Metric* scorer = NULL;
  DataPointDataset *training_set = NULL;
  DataPointDataset *validation_set = NULL;
  float training_score = 0.0f;
  float validation_bestscore = 0.0f;
  unsigned int partialsave_niterations = 0;
  char *output_basename = NULL;


 public:
  LTR_Algorithm() {
  }
  virtual ~LTR_Algorithm() {
    delete validation_set, delete training_set;
    free(output_basename);
  }

  //
  // TODO: Candidate class structure
  //
  // void learn (training, validation, metric, other params valid for all subclasses)
  //
  // score_type score_document (document)
  // score_type score_list (list of documents)
  // score_type score_dataset (collection of list of documents)
  //
  // save_model (filename)
  // load_model (filename)
  //
  // string get_name()
  //

  virtual float eval_dp(float * const * const features,
                        unsigned int idx) const = 0;  //prediction value to store in a file
  virtual const char *whoami() const = 0;
  virtual void showme() = 0;
  virtual void init() = 0;
  virtual void learn() = 0;
  virtual void write_outputtofile() = 0;

  void set_scorer(qr::metric::ir::Metric* ms) {
    scorer = ms;
  }
  void set_trainingset(DataPointDataset *trainingset) {
    training_set = trainingset;
  }
  void set_validationset(DataPointDataset *validationset) {
    validation_set = validationset;
  }
  void set_partialsave(unsigned int niterations) {
    partialsave_niterations = niterations;
  }
  void set_outputfilename(const char *filename) {
    output_basename = strdup(filename);
  }
  float compute_score(DataPointDataset *samples,
                      qr::metric::ir::Metric* scorer);
};

} // namespace learning
} // namespace quickrank

#endif
