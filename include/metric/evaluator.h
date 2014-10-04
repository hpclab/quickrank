#ifndef QUICKRANK_METRIC_EVALUATOR_H_
#define QUICKRANK_METRIC_EVALUATOR_H_

#include "metric/ir/metric.h"
#include "learning/ranker.h"

class evaluator {
  public:
    bool normalize = false;
  protected:
    LTR_Algorithm *r = NULL;
    qr::metric::ir::Metric* training_scorer = NULL;
    qr::metric::ir::Metric* test_scorer = NULL;
  public:
    evaluator(LTR_Algorithm *r,
              qr::metric::ir::Metric* training_scorer,
              qr::metric::ir::Metric* test_scorer) :
      r(r), training_scorer(training_scorer), test_scorer(test_scorer) {}
    ~evaluator();

    void evaluate(const char *trainingfilename, const char *validationfilename,
                  const char *testfilename, const char *featurefilename, const char *outputfilename);

    void write();
};

#endif

