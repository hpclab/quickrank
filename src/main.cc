#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <time.h>
#include <iostream>

// TODO: (by cla) Add some logging facility and remove any printf.
// TODO: (by cla) Give names to error codes.
// TODO: (by cla) It seems this log file is now useless ?!?
#ifdef LOGFILE
FILE *flog = NULL;
#endif

#define SHOWTIMER

#include "metric/evaluator.h"
#include "learning/lmart.h"
#include "learning/matrixnet.h"
#include "metric/ir/ndcg.h"
#include "metric/ir/map.h"

int main(int argc, char *argv[]) {
	//set seed for rand()
	srand(time(NULL));
	//index of current argv[]
	int argi = 1;
	//read ranker type and its parameters
	LTR_Algorithm *r = NULL;
	if(argi+6<argc && strcasecmp(argv[argi],"lm")==0) {
		unsigned int ntrees = atoi(argv[++argi]);
		float shrinkage = atof(argv[++argi]);
		unsigned int nthresholds = atoi(argv[++argi]);
		unsigned int ntreeleaves = atoi(argv[++argi]);
		unsigned int minleafsupport = atoi(argv[++argi]);
		unsigned int esr = atoi(argv[++argi]);
		r = new LambdaMart(ntrees, shrinkage, nthresholds, ntreeleaves, minleafsupport, esr), ++argi;
	} else if(argi+6<argc && strcasecmp(argv[argi],"mn")==0) {
		unsigned int ntrees = atoi(argv[++argi]);
		float shrinkage = atof(argv[++argi]);
		unsigned int nthresholds = atoi(argv[++argi]);
		unsigned int treedepth = atoi(argv[++argi]);
		unsigned int minleafsupport = atoi(argv[++argi]);
		unsigned int esr = atoi(argv[++argi]);
		r = new MatrixNet(ntrees, shrinkage, nthresholds, treedepth, minleafsupport, esr), ++argi;
	} else exit(11);
	//show ranker parameters
	printf("New ranker:\n");
	r->showme();

	//read metric scorer for the training phase and its parameters
	qr::metric::ir::Metric* training_scorer = NULL;
	if(argi+1<argc && strcasecmp(argv[argi],"ndcg")==0) {
		unsigned int k = atoi(argv[++argi]);
		training_scorer = new qr::metric::ir::Ndcg(k), ++argi;
	} else if(argi+1<argc && strcasecmp(argv[argi],"map")==0) {
		unsigned int k = atoi(argv[++argi]);
		training_scorer = new qr::metric::ir::Map(k), ++argi;
	} else exit(12);
	//show metric scorer parameters
	std::cout << "New training scorer: " << *training_scorer << std::endl;

	//read metric scorer for the test phase and its parameters
	qr::metric::ir::Metric* test_scorer = NULL;
	if(argi+1<argc && strcasecmp(argv[argi],"ndcg")==0) {
		unsigned int k = atoi(argv[++argi]);
		test_scorer = new qr::metric::ir::Ndcg(k), ++argi;
	} else if(argi+1<argc && strcasecmp(argv[argi],"map")==0) {
		unsigned int k = atoi(argv[++argi]);
		test_scorer = new qr::metric::ir::Map(k), ++argi;
	} else if(argi<argc && strcmp(argv[argi],"-")==0) {
		++argi;
	} else exit(13);
	//show test scorer parameters
	if(test_scorer)
	  std::cout << "New test scorer: " << *test_scorer << std::endl;
	else
	  std::cout << "New test scorer: (null)" << std::endl;

	//instantiate a new evaluator with read arguments
	evaluator ev(r, training_scorer, test_scorer);

	//set ranker partial save
	if(argi<argc) {
		unsigned int npartialsave = atoi(argv[argi++]);
		if(npartialsave>0)
			r->set_partialsave(npartialsave);
	} else exit(14);

	//read filenames to be passed to the evaluator from the last argv[]
	char *training_filename;
	char *validation_filename;
	char *test_filename;
	char *features_filename;
	char *output_basename;
	if(argi+4<argc) {
		printf("Filenames:\n");
		training_filename = strcmp(argv[argi],"-") ? argv[argi] : NULL;
		printf("\ttraining file = %s\n", training_filename), ++argi;
		validation_filename = strcmp(argv[argi],"-") ? argv[argi] : NULL;
		printf("\tvalidation file = %s\n", validation_filename), ++argi;
		test_filename = strcmp(argv[argi],"-") ? argv[argi] : NULL;
		printf("\ttest file = %s\n", test_filename), ++argi;
		features_filename = strcmp(argv[argi],"-") ? argv[argi] : NULL;
		printf("\tfeatures file = %s\n", features_filename), ++argi;
		output_basename = strcmp(argv[argi],"-") ? argv[argi] : NULL;
		printf("\toutput basename = %s\n", output_basename), ++argi;
	} else exit(15);

	//warnings for unexpected arguments
	if(argi!=argc)
		printf("warning: unexpected arguments\n");

	//start evaluation process
	ev.evaluate(training_filename, validation_filename, test_filename, features_filename, output_basename);
	if(output_basename)
		ev.write();

	return 0;
}
