#ifndef QUICKRANK_LEARNING_RANKER_H_
#define QUICKRANK_LEARNING_RANKER_H_

#include "data/ltrdata.h"
#include "data/dataset.h"
#include "utils/qsort.h"

#include "metric/ir/metric.h"

namespace quickrank {
namespace learning {

class LTR_Algorithm {



 protected:
  qr::metric::ir::Metric* scorer = NULL;
  LTR_VerticalDataset *training_set = NULL;
  LTR_VerticalDataset *validation_set = NULL;
  float training_score = 0.0f;
  float validation_bestscore = 0.0f;
  unsigned int partialsave_niterations = 0;
  char *output_basename = NULL;

  std::shared_ptr<quickrank::data::Dataset> validation_dataset;


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

  // TODO: to be moved ...
  virtual void score_dataset(quickrank::data::Dataset &dataset, qr::Score* scores) const {
    if (dataset.format()!=quickrank::data::Dataset::VERT)
      dataset.transpose();
    for (unsigned int q=0; q<dataset.num_queries(); q++) {
      std::shared_ptr<quickrank::data::QueryResults> r = dataset.getQueryResults(q);
      score_query_results(r, scores, dataset.num_instances());
      scores += r->num_results();
    }
  }
  // assumes vertical dataset
  // offset to next feature of the same instance
  virtual void score_query_results(std::shared_ptr<quickrank::data::QueryResults> results,
                                   qr::Score* scores,
                                   unsigned int offset) const {
    const qr::Feature* d = results->features();
    for (unsigned int i=0; i<results->num_results(); i++) {
      scores[i] = score_document(d,offset);
      d++;
    }
  }
  // assumes vertical dataset
  virtual qr::Score score_document(const qr::Feature* d, const unsigned int offset=1) const {
    return 0.0;
  }

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
  void set_trainingset(LTR_VerticalDataset *trainingset) {
    training_set = trainingset;
  }
  void set_validationset(LTR_VerticalDataset *validationset) {
    validation_set = validationset;
  }
  void set_validation_dataset(std::shared_ptr<quickrank::data::Dataset> d) {
    validation_dataset = d;
  }
  void set_partialsave(unsigned int niterations) {
    partialsave_niterations = niterations;
  }
  void set_outputfilename(const char *filename) {
    output_basename = strdup(filename);
  }
  float compute_score(LTR_VerticalDataset *samples,
                      qr::metric::ir::Metric* scorer);
};

} // namespace learning
} // namespace quickrank

#endif
