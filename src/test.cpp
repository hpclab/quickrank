#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <cfloat>
#include <limits>

#define SHOWTIMER

#include "utils/strutils.hpp"
#include "learning/dpset.hpp"

#define TRAINFILENAME "../datasets/rand/train.txt"//"/media/gabriele/kapasdhc/Fold0/train.txt" //"/media/gabriele/kapasdhc/Fold0/train.txt" //

int main(int argc, char *argv[]) {
	char *filename[4] = {NULL};
	for(int counter=0; counter<4 and is_empty(argv[1])==false; ++counter) {
		char *s = read_token(argv[1],',');
		printf("[%s]\n", s);
		if(*s!=',')
			filename[counter] = s;
		else 
			++argv[1];
	}
	for(int i=0; i<4; ++i)
		printf("<%s>\n", filename[i]);
	return 0;
}
