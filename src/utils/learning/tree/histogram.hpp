#ifndef __HISTOGRAM_HPP__
#define __HISTOGRAM_HPP__

#include <cmath> //NAN

static_assert(sizeof(unsigned int)==4,"sizeof(unsigned int) exception!");

#include "learning/dpset.hpp"

struct splitparams {
	unsigned int best_featureid = 0xFFFFFFFF;
	float best_threshold = NAN;
	float deviance = NAN;
	float lvar = NAN;
	float lsum = NAN;
	unsigned int lcount = 0;
	float rvar = NAN;
	float rsum = NAN;
	unsigned int rcount = 0;
};

class histogram {
	public:
		float **thresholds = NULL; //[0..nfeatures-1]x[0..thresholds_size[i]-1]
		unsigned int *thresholds_size = NULL;
		unsigned int **stmap = NULL; //[0..nfeatures-1]x[0..nthresholds-1] //TODO sparse array... maybe is possible to reimplement it
		unsigned int nfeatures = 0;
		float samplingrate = 1.0f;
		float **sumlbl = NULL; //[0..nfeatures-1]x[0..nthresholds-1]
		float **sqsumlbl = NULL; //[0..nfeatures-1]x[0..nthresholds-1]
		unsigned int **count = NULL; //[0..nfeatures-1]x[0..nthresholds-1]
	public:
		histogram(float **thresholds, unsigned int *thresholds_size, unsigned int nfeatures) : thresholds(thresholds), thresholds_size(thresholds_size), nfeatures(nfeatures) {
			sumlbl = new float*[nfeatures],
			sqsumlbl = new float*[nfeatures],
			count = new unsigned int*[nfeatures];
			for(unsigned int i=0; i<nfeatures; ++i) {
				const unsigned int threshold_size = thresholds_size[i];
				sumlbl[i] = new float[threshold_size]();
				sqsumlbl[i] = new float[threshold_size]();
				count[i] = new unsigned int[threshold_size]();
			}
		}
		~histogram() {
			for(unsigned int i=0; i<nfeatures; ++i)
					delete [] sumlbl[i],
					delete [] sqsumlbl[i],
					delete [] count[i];
			delete [] sumlbl,
			delete [] sqsumlbl,
			delete [] count;
		}
		void update(float *labels, const unsigned int nlabels) {
			#pragma omp parallel for
			for(unsigned int i=0; i<nfeatures; ++i)
				for(unsigned int t=0; t<thresholds_size[i]; ++t)
					sumlbl[i][t] = 0.0f,
					sqsumlbl[i][t] = 0.0f;
			#pragma omp parallel for
			for(unsigned int i=0; i<nfeatures; ++i)
				for(unsigned int k=0; k<nlabels; ++k) {
					const unsigned int t = stmap[i][k];
					sumlbl[i][t] += labels[k],
					sqsumlbl[i][t] += labels[k]*labels[k];
					//count doesn't change, so no need to re-compute
				}
			#pragma omp parallel for
			for(unsigned int i=0; i<nfeatures; ++i)
				for(unsigned int t=1; t<thresholds_size[i]; ++t)
					sumlbl[i][t] += sumlbl[i][t-1],
					sqsumlbl[i][t] += sqsumlbl[i][t-1];
		}
		splitparams *find_bestsplit(const unsigned int minls, const float initvar=FLT_MAX) {
			//featureidxs to be used for tree splitnodeting
			unsigned int featuresamples[nfeatures];
			unsigned int nfeaturesamples = 0;
			for(unsigned int i=0; i<nfeatures; ++i)
				featuresamples[nfeaturesamples++] = i;
			if(samplingrate<1.0f) {
				//need to make a sub-sampling
				unsigned int reduced_nfeaturesamples = (unsigned int)floor(samplingrate*nfeaturesamples);
				while(nfeaturesamples>reduced_nfeaturesamples && nfeaturesamples>1) {
					unsigned int selected = rand()%nfeaturesamples;
					featuresamples[selected] = featuresamples[--nfeaturesamples];
				}
			}
			//find best rtnode
			unsigned int best_featureid = 0xFFFFFFFF;
			unsigned int best_thresholdid = 0xFFFFFFFF;
			float best_lvar = 0.0f;
			float best_rvar = 0.0f;
			float minvar = initvar;
			for(unsigned int i=0; i<nfeaturesamples; ++i) {
				const unsigned int f = featuresamples[i];
				//define pointer shortcuts
				float *sumlabels = sumlbl[f];
				float *sqsumlabels = sqsumlbl[f];
				unsigned int *samplecount = count[f];
				//get last elements
				unsigned int threshold_size = thresholds_size[f];
				float s = sumlabels[threshold_size-1];
				float sq = sqsumlabels[threshold_size-1];
				unsigned int c = samplecount[threshold_size-1];
				//looking for the feature that minimizes sum of lvar+rvar
				for(unsigned int t=0; t<threshold_size; ++t) {
					unsigned int lcount = samplecount[t];
					unsigned int rcount = c-lcount;
					if(lcount>=minls && rcount>=minls)  {
						float lsum = sumlabels[t];
						float lsqsum = sqsumlabels[t];
						float lvar = fabs(lsqsum-lsum*lsum/lcount);
						float rsum = s-lsum;
						float rsqsum = sq-lsqsum;
						float rvar = fabs(rsqsum-rsum*rsum/rcount);
						float sumvar = lvar+rvar;
						if(FLT_EPSILON+sumvar<minvar) //is required at least an FLT_EPSILON improvment
							minvar = sumvar,
							best_lvar = lvar,
							best_rvar = rvar,
							best_featureid = f,
							best_thresholdid = t;
					}
				}
			}

			if(minvar==initvar) return NULL; //unsplitable

			unsigned int last_thresholdidx = thresholds_size[best_featureid]-1;

			splitparams *result = new splitparams();
			result->best_featureid = best_featureid,
			result->best_threshold = thresholds[best_featureid][best_thresholdid],
			result->deviance = minvar,
			result->lvar = best_lvar,
			result->lsum = sumlbl[best_featureid][best_thresholdid],
			result->lcount = count[best_featureid][best_thresholdid];
			result->rvar = best_rvar,
			result->rsum = sumlbl[best_featureid][last_thresholdidx]-result->lsum,
			result->rcount = count[best_featureid][last_thresholdidx]-result->lcount;
			return result;

		}
};

class temphistogram : public histogram {
	public:
		temphistogram(histogram const* parent, unsigned int const* sampleids, const unsigned int nsampleids, float const* labels) : histogram(parent->thresholds, parent->thresholds_size, parent->nfeatures) {
			stmap = parent->stmap;
			#pragma omp parallel for
			for(unsigned int i=0; i<nfeatures; ++i) {
				for(unsigned int j=0; j<nsampleids; ++j) {
					const unsigned int k = sampleids[j];
					const unsigned int t = stmap[i][k];
					sumlbl[i][t] += labels[k],
					sqsumlbl[i][t] += labels[k]*labels[k],
					count[i][t]++;
				}
				for(unsigned int t=1; t<thresholds_size[i]; ++t)
					sumlbl[i][t] += sumlbl[i][t-1],
					sqsumlbl[i][t] += sqsumlbl[i][t-1],
					count[i][t] += count[i][t-1];
			}
		}
		temphistogram(histogram const* parent, histogram const* left) : histogram(parent->thresholds, parent->thresholds_size, parent->nfeatures) {
			stmap = parent->stmap;
			#pragma omp parallel for
			for(unsigned int i=0; i<nfeatures; ++i) {
				const unsigned int nthresholds = thresholds_size[i];
				for(unsigned int t=0; t<nthresholds; ++t)
					sumlbl[i][t] = parent->sumlbl[i][t]-left->sumlbl[i][t],
					sqsumlbl[i][t] += parent->sqsumlbl[i][t]-left->sqsumlbl[i][t],
					count[i][t] += parent->count[i][t]-left->count[i][t];
			}
		}
};

class permhistogram : public histogram {
	public:
		permhistogram(dpset *dps, float *labels, unsigned int **sortedidx, unsigned int *sortedidxsize, float **thresholds, unsigned int *thresholds_size) : histogram(thresholds, thresholds_size, dps->get_nfeatures()) {
			stmap = new unsigned int*[nfeatures];
			#pragma omp parallel for
			for(unsigned int i=0; i<nfeatures; ++i) {
				stmap[i] = new unsigned int[sortedidxsize[i]];
				unsigned int threshold_size = thresholds_size[i];
				float *features = dps->get_fvector(i);
				float *threshold = thresholds[i];
				float sum = 0.0f;
				float sqsum = 0.0f;
				for(unsigned int last=-1, j, t=0; t<threshold_size; ++t) {
					//find the first sample exceeding the current threshold
					for(j=last+1; j<sortedidxsize[i]; ++j) {
						unsigned int k = sortedidx[i][j];
						if(features[k]>threshold[t]) break;
						sum += labels[k],
						sqsum += labels[k]*labels[k],
						stmap[i][k] = t;
					}
					last = j-1,
					sumlbl[i][t] = sum,
					sqsumlbl[i][t] = sqsum,
					count[i][t] = j;
				}
			}
		}
		~permhistogram() {
			for(unsigned int i=0; i<nfeatures; ++i)
				delete [] stmap[i];
			delete [] stmap;
		}
};

#endif
