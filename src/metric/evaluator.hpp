#ifndef __EVALUATOR_HPP__
#define __EVALUATOR_HPP__

#include "metric/mapscorer.hpp"
#include "metric/ndcgscorer.hpp"
#include "metric/dcgscorer.hpp"
#include "learning/lmart.hpp"
//#include "learning/matrixnet.hpp"

class evaluator {
	public:
		bool normalize = false;
	protected:
		ranker *r = NULL;
		metricscorer *training_scorer = NULL;
		metricscorer *test_scorer = NULL;
	public:
		evaluator(ranker *r, metricscorer *training_scorer, metricscorer *test_scorer):
			r(r),
			training_scorer(training_scorer),
			test_scorer(test_scorer) {
			printf("> NEW EVALUATOR:\n\tranker type = %s\n\ttrain scorer type = %s\n\ttest scorer type = %s\n", r->whoami(), training_scorer->whoami(), test_scorer?test_scorer->whoami():"not required");
		}
		~evaluator() {
			delete r;
			delete training_scorer;
			delete test_scorer;
		}
		void evaluate(const char *trainingfilename, const char *validationfilename, const char *testfilename, const char *featurefilename) {
			if(not is_empty(trainingfilename)) {
				printf(">> TRAINING DATASET:\n");
				r->set_trainingset(new dpset(trainingfilename));
			} else return;
			if(not is_empty(validationfilename)) {
				printf(">> VALIDATION DATASET:\n");
				r->set_validationset(new dpset(validationfilename));
			}
			dpset *testset = NULL;
			if(not is_empty(testfilename) and test_scorer) {
				printf(">> TEST DATASET:\n");
				testset = new dpset(testfilename);
			}
			if(not is_empty(featurefilename)) {
				// init featureidxs from file
			}
			if(normalize) {
				//normalization
			}
			r->set_scorer(training_scorer);
			r->init();
			r->learn();
			r->write_outputtofile("/tmp/output.xml");
			if(testset) {
				printf(">>> TESTING:\n\t%s@%u on test data = %.4f\n", test_scorer->whoami(), test_scorer->get_k(), r->compute_score(testset, test_scorer));
				delete testset;
			}
		}
};

#endif
