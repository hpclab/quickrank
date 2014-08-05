#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <time.h>

#define SHOWTIMER

#include "metric/evaluator.hpp"
#include "learning/ranker.hpp"

int main(int argc, char *argv[]) {
	//set seed for rand()
	srand(time(NULL));
	//index of current argv[]
	int argi = 1;
	//read ranker type and its parameters
	ranker *r = NULL;
	if(argi+6<argc && strcmp(argv[argi],"lm")==0) {
		unsigned int ntrees = atoi(argv[++argi]);
		float shrinkage = atof(argv[++argi]);
		unsigned int nthresholds = atoi(argv[++argi]);
		unsigned int ntreeleaves = atoi(argv[++argi]);
		unsigned int minleafsupport = atoi(argv[++argi]);
		unsigned int esr = atoi(argv[++argi]);
		r = new lmart(ntrees, shrinkage, nthresholds, ntreeleaves, minleafsupport, esr), ++argi;
	} else if(argi+6<argc && strcmp(argv[argi],"mn")==0) {
		unsigned int ntrees = atoi(argv[++argi]);
		float shrinkage = atof(argv[++argi]);
		unsigned int nthresholds = atoi(argv[++argi]);
		unsigned int treedepth = atoi(argv[++argi]);
		unsigned int minleafsupport = atoi(argv[++argi]);
		unsigned int esr = atoi(argv[++argi]);
		r = new matrixnet(ntrees, shrinkage, nthresholds, treedepth, minleafsupport, esr), ++argi;
	} else exit(11);
	//show ranker parameters
	printf("New ranker:\n");
	r->showme();

	//read metric scorer for the training phase and its parameters
	metricscorer *training_scorer = NULL;
	if(argi+1<argc && strcmp(argv[argi],"ndcg")==0) {
		unsigned int k = atoi(argv[++argi]);
		training_scorer = new ndcgscorer(k), ++argi;
	} else if(argi+1<argc && strcmp(argv[argi],"map")==0) {
		unsigned int k = atoi(argv[++argi]);
		training_scorer = new mapscorer(k), ++argi;
	} else exit(12);
	//show metric scorer parameters
	printf("New training scorer:\n");
	training_scorer->showme();

	//read metric scorer for the test phase and its parameters
	metricscorer *test_scorer = NULL;
	if(argi+1<argc && strcmp(argv[argi],"ndcg")==0) {
		unsigned int k = atoi(argv[++argi]);
		test_scorer = new ndcgscorer(k), ++argi;
	} else if(argi+1<argc && strcmp(argv[argi],"map")==0) {
		unsigned int k = atoi(argv[++argi]);
		test_scorer = new mapscorer(k), ++argi;
	} else if(argi<argc && strcmp(argv[argi],"-")==0) {
		++argi;
	} else exit(13);
	//show test scorer parameters
	printf("New test scorer:\n");
	if(test_scorer) test_scorer->showme();
	else printf("\t(null)\n");

	//instantiate a new evaluator with read arguments
	evaluator ev(r, training_scorer, test_scorer);

	//set partial save
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
		printf("\ttraining file = '%s'\n", training_filename), ++argi;
		validation_filename = strcmp(argv[argi],"-") ? argv[argi] : NULL;
		printf("\tvalidation file = '%s'\n", validation_filename), ++argi;
		test_filename = strcmp(argv[argi],"-") ? argv[argi] : NULL;
		printf("\ttest file = '%s'\n", test_filename), ++argi;
		features_filename = strcmp(argv[argi],"-") ? argv[argi] : NULL;
		printf("\tfeatures file = '%s'\n", features_filename), ++argi;
		output_basename = strcmp(argv[argi],"-") ? argv[argi] : NULL;
		printf("\toutput basename = '%s'\n", output_basename), ++argi;
		ev.evaluate(training_filename, validation_filename, test_filename, features_filename, output_basename);
		if(output_basename)
			ev.write();
	} else exit(15);

	//warnings for unexpected arguments
	if(argi!=argc)
		printf("warning: unexpected arguments\n");
	return 0;
}
