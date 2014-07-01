#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <cfloat>
#include <limits>

#define SHOWTIMER

#include "learning/dpset.hpp"

#define TRAINFILENAME "../datasets/rand/train.txt"//"/media/gabriele/kapasdhc/Fold0/train.txt" //"/media/gabriele/kapasdhc/Fold0/train.txt" //

int main() {
	srand(time(NULL));
	dpset dps(TRAINFILENAME);
	return 0;
}
