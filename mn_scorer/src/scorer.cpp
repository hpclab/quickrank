
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <omp.h>

#define SHL(n,p) ((n)<<(p))

#include "dpset.hpp"
#include "scorer.hpp"

#define FEATURE_PADDING 1 //previous models.xml had feature-indexs instead of feature-id, this value re-align id and indexes

unsigned int eval_dpontree(float *dpfeatures, unsigned int const *featureidxs, float const *thresholds, const int m) {
	unsigned int leafidx = 0;
	for(int i=0; i<M; ++i)
		leafidx |= SHL(dpfeatures[featureidxs[i]+FEATURE_PADDING]>thresholds[i] && i<m, M-1-i);
	return leafidx;
}

float eval_dp(float *fvector) {
	float score = 0.0f;
	#pragma omp parallel for reduction(+:score)
	for(int i=0; i<N; ++i)
		score += ws[i] * os[i][eval_dpontree(fvector, fs[i], ts[i], ds[i])];
	return score;
}

int main(int argc, char *argv[]) {
	//check no of arguments
	if(argc!=3) {
		printf("\nUsage: %s data_test_filename score_filename\n\n", argv[0]);
		return 1;
	}
	//read datapoints
	dpset dps(argv[1]);
	//create score array
	const unsigned int ndps = dps.get_ndatapoints();
	float *scores = new float[ndps];
	//start timer
	double processingtime = omp_get_wtime();
	//compute score
	for(unsigned int i=0; i<ndps; ++i)
		scores[i] = eval_dp(dps.get_fvector(i));
	//stop timer
	processingtime = omp_get_wtime()-processingtime;
	//print time
	printf("%e\t", processingtime);
	//reorder datapoint scores wrt datapoint positions in the dataset
	float *sorted_scores = new float[ndps];
	for(unsigned int i=0; i<ndps; ++i)
		sorted_scores[dps.get_nline(i)] = scores[i];
	//write scores to file
	FILE *outf = fopen(argv[2], "w");
	if(outf) {
		for(unsigned int i=0; i<ndps; ++i)
			fprintf(outf, "%+.4f\n", sorted_scores[i]);
		fclose(outf);
		printf("%s\n", argv[2]);
	}
	//free mem
	delete [] scores,
	delete [] sorted_scores;
	return 0;
}
