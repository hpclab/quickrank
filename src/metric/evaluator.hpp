#ifndef __EVALUATOR_HPP__
#define __EVALUATOR_HPP__

#include "metric/mapscorer.hpp"
#include "metric/ndcgscorer.hpp"
#include "metric/dcgscorer.hpp"
#include "learning/lmart.hpp"
#include "learning/matrixnet.hpp"

class evaluator {
	public:
		bool normalize = false;
	protected:
		ranker *r = NULL;
		metricscorer *training_scorer = NULL;
		metricscorer *test_scorer = NULL;
	public:
		evaluator(ranker *r, metricscorer *training_scorer, metricscorer *test_scorer): r(r), training_scorer(training_scorer), test_scorer(test_scorer) {}
		~evaluator() {
			// TODO: (by cla) Do not delete objected you didn't create. Move.
			delete r;
			delete training_scorer;
			delete test_scorer;
		}
		void evaluate(const char *trainingfilename, const char *validationfilename, const char *testfilename, const char *featurefilename, const char *outputfilename) {
			if(not is_empty(trainingfilename)) {
				printf("Reading Training dataset:\n");
				// TODO: (by cla) Where is the delete of this dpset?
				r->set_trainingset(new dpset(trainingfilename));
			} else exit(6);
			if(not is_empty(validationfilename)) {
				// TODO: (by cla) Where is the delete of this dpset?
				printf("Reading validation dataset:\n");
				r->set_validationset(new dpset(validationfilename));
			}
			dpset *testset = NULL;
			if(not is_empty(testfilename) and test_scorer) {
				printf("Reading test dataset:\n");
				testset = new dpset(testfilename);
			}
			if(not is_empty(featurefilename)) {
				// init featureidxs from file
			}
			if(not is_empty(outputfilename))
				r->set_outputfilename(outputfilename);
			if(normalize) {
				//normalization
			}
			r->set_scorer(training_scorer);
			r->init();
			r->learn();
			if(testset) {
				printf("Testing:\n");
				#ifdef SHOWTIMER
				double timer = omp_get_wtime();
				#endif
				float score = r->compute_score(testset, test_scorer);
				#ifdef SHOWTIMER
				timer = omp_get_wtime()-timer;
				#endif
				printf("\t%s@%u on test data = %.4f\n", test_scorer->whoami(), test_scorer->get_k(), score);
				#ifdef SHOWTIMER
				printf("\telapsed time = %.3f seconds\n", timer);
				#endif
				printf("\tdone\n");
				delete testset;
			}
		}
		void write() {
			printf("Writing output:\n");
			r->write_outputtofile();
			printf("\tdone\n");
		}
};

#endif
