#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <cfloat>
#include <limits>

#define SHOWTIMER

#include "utils/listqsort.hpp"
#include "learning/dpset.hpp"

#define TRAINFILENAME "../datasets/rand/train.txt"//"/media/gabriele/kapasdhc/Fold0/train.txt" //"/media/gabriele/kapasdhc/Fold0/train.txt" //

int main() {
	srand(time(NULL));
	dpset s(TRAINFILENAME);
	unsigned int n = s.get_nrankedlists();
	for(unsigned int i=0; i<n; ++i) {
		rnklst rl = s.get_ranklist(i);
		printf("%u id=%u\n", i, rl.id);
		for(unsigned int j=0; j<rl.size; ++j)
			printf("   %u %f\n", j, rl.labels[j]);
	}
	return 0;
}
