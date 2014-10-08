#ifndef QUICKRANK_METRIC_EVALUATOR_H_
#define QUICKRANK_METRIC_EVALUATOR_H_

#include "metric/ir/metric.h"
#include "learning/ltr_algorithm.h"

namespace qr {
namespace metric {

class evaluator : private boost::noncopyable {
  public:
    bool normalize = false;
  protected:
    quickrank::learning::LTR_Algorithm *r = NULL;
    qr::metric::ir::Metric* training_scorer = NULL;
    qr::metric::ir::Metric* test_scorer = NULL;
  public:
    evaluator(quickrank::learning::LTR_Algorithm *r,
              qr::metric::ir::Metric* training_scorer,
              qr::metric::ir::Metric* test_scorer) :

      r(r), training_scorer(training_scorer), test_scorer(test_scorer) {}
    ~evaluator();

    // TODO: Remove to support just std::string version
    void evaluate(const char *trainingfilename, const char *validationfilename,
                  const char *testfilename, const char *featurefilename, const char *outputfilename);

    void evaluate(const std::string &trainingfilename,
    		      const std::string &validationfilename,
                  const std::string &testfilename,
				  const std::string &featurefilename,
				  const std::string &outputfilename)
    {
    	evaluate(trainingfilename.c_str(),
    			 validationfilename.c_str(),
				 testfilename.c_str(),
				 featurefilename.c_str(),
				 outputfilename.c_str());
    }

    void write();
};

} // namespace metric
} // namespace qr

#endif

