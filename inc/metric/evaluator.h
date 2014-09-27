#ifndef QUICKRANK_METRIC_EVALUATOR_H_
#define QUICKRANK_METRIC_EVALUATOR_H_

#include "metric/mapscorer.h"
#include "metric/ndcgscorer.h"
#include "metric/dcgscorer.h"
#include "learning/lmart.h"
#include "learning/matrixnet.h"

class evaluator {
  public:
    bool normalize = false;
  protected:
    Ranker *r = NULL;
    MetricScorer *training_scorer = NULL;
    MetricScorer *test_scorer = NULL;
  public:
    evaluator(Ranker *r, MetricScorer *training_scorer, MetricScorer *test_scorer) :
      r(r), training_scorer(training_scorer), test_scorer(test_scorer) {}
    ~evaluator();

    void evaluate(const char *trainingfilename, const char *validationfilename,
                  const char *testfilename, const char *featurefilename, const char *outputfilename);

    void write();
};

#endif

