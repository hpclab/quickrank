#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <cfloat>
#include <limits>
#include <cmath> // NAN, isnan()
#include <omp.h> // omp_get_thread_num()

#include "learning/dpset.hpp"

#define TRAINFILENAME "/tmp/train.txt"//"/media/gabriele/kapasdhc/Fold0/train.txt" //"/media/gabriele/kapasdhc/Fold0/train.txt" //

int main() {
	srand(time(NULL));
	dpset dps(TRAINFILENAME);
	//dps.write("/tmp/pippo.txt");
	return 0;
}
