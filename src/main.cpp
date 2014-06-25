#include <cstdio>
#include <cstdlib>
#include <cfloat>
#include <time.h>

//#define LOGFILE
#define SHOWTIMER

#ifdef LOGFILE
FILE *flog = NULL;
#endif

#include "metric/evaluator.hpp"
#include "learning/ranker.hpp"

#define TRAINFILENAME "../datasets/rand/train.txt"
#define VALIDATIONFILENAME "../datasets/rand/vali.txt"
#define TESTFILENAME "../datasets/rand/test.txt"

int main(int argc, char *argv[]) {
	#ifdef LOGFILE
	flog = fopen("/tmp/ranklib.log", "w");
	#endif
	srand(time(NULL));
	evaluator ev(new lmartranker(), new ndcgscorer(10), new ndcgscorer(3));
	//evaluator ev(new lmartranker(), new ndcgscorer(10), NULL);
	ev.evaluate(TRAINFILENAME, VALIDATIONFILENAME, TESTFILENAME, "");
	//ev.evaluate(TRAINFILENAME, "", "", "");
	#ifdef LOGFILE
	fclose(flog);
	#endif
	return 0;
}
