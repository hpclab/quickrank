#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <time.h>

//#define LOGFILE
//#define SHOWTIMER

#ifdef LOGFILE
FILE *flog = NULL;
#endif

#include "metric/evaluator.hpp"
#include "learning/ranker.hpp"

int main(int argc, char *argv[]) {
	srand(time(NULL));
	#ifdef LOGFILE
	flog = fopen("/tmp/quickrank.log", "w");
	#endif
	//index of current argv[]
	int argi = 1;
	//read ranker type and its parameters
	ranker *r = NULL;
	printf("New ranker:\n");
	if(argi+6<argc && strcmp(argv[argi],"lm")==0) {
		unsigned int ntrees = atoi(argv[++argi]);
		float learningrate = atof(argv[++argi]);
		unsigned int nthresholds = atoi(argv[++argi]);
		unsigned int ntreeleaves = atoi(argv[++argi]);
		unsigned int minleafsupport = atoi(argv[++argi]);
		unsigned int esr = atoi(argv[++argi]);
		r = new lmart(ntrees, learningrate, nthresholds, ntreeleaves, minleafsupport, esr), ++argi;
	} else if(argi+6<argc && strcmp(argv[argi],"mn")==0) {
		unsigned int ntrees = atoi(argv[++argi]);
		float learningrate = atof(argv[++argi]);
		unsigned int nthresholds = atoi(argv[++argi]);
		unsigned int treedepth = atoi(argv[++argi]);
		unsigned int minleafsupport = atoi(argv[++argi]);
		unsigned int esr = atoi(argv[++argi]);
		r = new matrixnet(ntrees, learningrate, nthresholds, treedepth, minleafsupport, esr), ++argi;
	} else exit(16);
	//read metric scorer for the training phase and its parameters
	metricscorer *training_scorer = NULL;
	printf("New training scorer:\n");
	if(argi+1<argc && strcmp(argv[argi],"NDCG")==0) {
		unsigned int k = atoi(argv[++argi]);
		training_scorer = new ndcgscorer(k), ++argi;
	} else if(argi+1<argc && strcmp(argv[argi],"MAP")==0) {
		unsigned int k = atoi(argv[++argi]);
		training_scorer = new mapscorer(k), ++argi;
	} else exit(17);
	//read metric scorer for the test phase and its parameters
	metricscorer *test_scorer = NULL;
	printf("New test scorer:\n");
	if(argi+1<argc && strcmp(argv[argi],"NDCG")==0) {
		unsigned int k = atoi(argv[++argi]);
		test_scorer = new ndcgscorer(k), ++argi;
	} else if(argi+1<argc && strcmp(argv[argi],"MAP")==0) {
		unsigned int k = atoi(argv[++argi]);
		training_scorer = new mapscorer(k), ++argi;
	} else if(argi<argc && strcmp(argv[argi],"-")==0) {
		printf("\t(skipped)\n"), ++argi;
	}else exit(18);
	//instantiate a new evaluator with read arguments
	evaluator ev(r, training_scorer, test_scorer);
	//read filenames to be passed to the evaluator from the last argv[]
	if(argi+3<argc) {
		char *training_filename = strcmp(argv[argi+0],"-") ? argv[argi+0] : NULL;
		char *validation_filename = strcmp(argv[argi+1],"-") ? argv[argi+1] : NULL;
		char *test_filename = strcmp(argv[argi+2],"-") ? argv[argi+2] : NULL;
		char *features_filename = strcmp(argv[argi+3],"-") ? argv[argi+3] : NULL;
		printf("Filenames:\n\ttraining_file = %s\n\tvalidation_file = %s\n\ttest_file = %s\n\tfeatures_file = %s\n", training_filename, validation_filename, test_filename, features_filename);
		ev.evaluate(training_filename, validation_filename, test_filename, features_filename);
	} else exit(19);
	//warnings for unexpected arguments
	if(argi+4!=argc)
		printf("warning: unexpected arguments\n");
	#ifdef LOGFILE
	fclose(flog);
	#endif
	return 0;
}
