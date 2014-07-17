#include <cstdio>
#include <cstdlib>
#include <cfloat>
#include <time.h>

//#define LOGFILE
//#define SHOWTIMER

#ifdef LOGFILE
FILE *flog = NULL;
#endif

#include "metric/evaluator.hpp"
#include "learning/ranker.hpp"

#define TRAINFILENAME "/data/gabriele/letor-tois/Fold1/train.txt" //"../datasets/MQ2008/Fold1/train.txt" //
#define VALIDATIONFILENAME "/data/gabriele/letor-tois/Fold1/vali.txt" //"../datasets/MQ2008/Fold1/vali.txt" //
#define TESTFILENAME "/data/gabriele/letor-tois/Fold1/test.txt" //"../datasets/MQ2008/Fold1/test.txt" //

int main(int argc, char *argv[]) {
	#ifdef LOGFILE
	flog = fopen("/tmp/quickrank.log", "w");
	#endif
	srand(time(NULL));
	evaluator ev(new lmart(), new ndcgscorer(10), new ndcgscorer(10));
	ev.evaluate(TRAINFILENAME, VALIDATIONFILENAME, TESTFILENAME, "");
	//ev.evaluate(TRAINFILENAME, "", "", "");
	#ifdef LOGFILE
	fclose(flog);
	#endif
	return 0;
}
