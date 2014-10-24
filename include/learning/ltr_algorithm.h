#ifndef QUICKRANK_LEARNING_RANKER_H_
#define QUICKRANK_LEARNING_RANKER_H_

#include "data/ltrdata.h"
#include "data/dataset.h"
#include "utils/qsort.h"

#include "metric/ir/metric.h"

namespace quickrank {
namespace learning {

class LTR_Algorithm {

 public:

 protected:

 private:
  /// The output stream operator.
  friend std::ostream& operator<<(std::ostream& os, const LTR_Algorithm& a) {
    return a.put(os);
  }
  /// Prints the description of Algorithm, including its parameters
  virtual std::ostream& put(std::ostream& os) const = 0;

 protected:
  quickrank::metric::ir::Metric* scorer = NULL;

  unsigned int partialsave_niterations = 0;
  std::string output_basename;

  std::shared_ptr<quickrank::data::Dataset> validation_dataset;
  std::shared_ptr<quickrank::data::Dataset> training_dataset;

 public:
  LTR_Algorithm() {
  }

  virtual ~LTR_Algorithm() {
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
  //

  virtual void score_dataset(quickrank::data::Dataset &dataset,
                             quickrank::Score* scores) const {
    if (dataset.format() != quickrank::data::Dataset::VERT)
      dataset.transpose();
    for (unsigned int q = 0; q < dataset.num_queries(); q++) {
      std::shared_ptr<quickrank::data::QueryResults> r =
          dataset.getQueryResults(q);
      score_query_results(r, scores, dataset.num_instances());
      scores += r->num_results();
    }
  }
  // assumes vertical dataset
  // offset to next feature of the same instance
  virtual void score_query_results(
      std::shared_ptr<quickrank::data::QueryResults> results,
      quickrank::Score* scores, unsigned int offset) const {
    const quickrank::Feature* d = results->features();
    for (unsigned int i = 0; i < results->num_results(); i++) {
      scores[i] = score_document(d, offset);
      d++;
    }
  }
  // assumes vertical dataset
  virtual quickrank::Score score_document(const quickrank::Feature* d,
                                          const unsigned int offset = 1) const {
    return 0.0;
  }

  virtual float eval_dp(float * const * const features,
                        unsigned int idx) const = 0;  //prediction value to store in a file

  virtual void init() = 0;
  virtual void learn() = 0;
  virtual void write_outputtofile() = 0;

  void set_scorer(quickrank::metric::ir::Metric* ms) {
    scorer = ms;
  }
  void set_training_dataset(std::shared_ptr<quickrank::data::Dataset> d) {
    training_dataset = d;
  }
  void set_validation_dataset(std::shared_ptr<quickrank::data::Dataset> d) {
    validation_dataset = d;
  }
  void set_partialsave(unsigned int niterations) {
    partialsave_niterations = niterations;
  }
  void set_outputfilename(const std::string filename) {
    output_basename = filename;
  }
  float compute_score(LTR_VerticalDataset *samples,
                      quickrank::metric::ir::Metric* scorer);
};

}  // namespace learning
}  // namespace quickrank

#endif
