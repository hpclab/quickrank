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
    LTR_Algorithm *r = NULL;
    Metric *training_scorer = NULL;
    Metric *test_scorer = NULL;
  public:
    evaluator(LTR_Algorithm *r, Metric *training_scorer, Metric *test_scorer) :
      r(r), training_scorer(training_scorer), test_scorer(test_scorer) {}
    ~evaluator();

    void evaluate(const char *trainingfilename, const char *validationfilename,
                  const char *testfilename, const char *featurefilename, const char *outputfilename);

    void write();
};

#endif

