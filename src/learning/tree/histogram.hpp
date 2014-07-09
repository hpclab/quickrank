#ifndef __HISTOGRAM_HPP__
#define __HISTOGRAM_HPP__

#include "learning/dpset.hpp"

struct histogram {
		float **thresholds = NULL; //[0..nfeatures-1]x[0..thresholds_size[i]-1]
		unsigned int const *thresholds_size = NULL;
		unsigned int **stmap = NULL; //[0..nfeatures-1]x[0..nthresholds-1] //TODO sparse array... maybe is possible to reimplement it
		unsigned int nfeatures = 0;
		float samplingrate = 1.0f;
		float **sumlbl = NULL; //[0..nfeatures-1]x[0..nthresholds-1]
		float **sqsumlbl = NULL; //[0..nfeatures-1]x[0..nthresholds-1]
		unsigned int **count = NULL; //[0..nfeatures-1]x[0..nthresholds-1]
		histogram(float **thresholds, unsigned int const *thresholds_size, unsigned int nfeatures) : thresholds(thresholds), thresholds_size(thresholds_size), nfeatures(nfeatures) {
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
};

struct temphistogram : public histogram {
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

struct permhistogram : public histogram {
	public:
		permhistogram(dpset *dps, float *labels, unsigned int **sortedidx, unsigned int sortedidxsize, float **thresholds, unsigned int const *thresholds_size) : histogram(thresholds, thresholds_size, dps->get_nfeatures()) {
			stmap = new unsigned int*[nfeatures];
			#pragma omp parallel for
			for(unsigned int i=0; i<nfeatures; ++i) {
				stmap[i] = new unsigned int[sortedidxsize];
				unsigned int threshold_size = thresholds_size[i];
				float *features = dps->get_fvector(i);
				float *threshold = thresholds[i];
				float sum = 0.0f;
				float sqsum = 0.0f;
				for(unsigned int last=-1, j, t=0; t<threshold_size; ++t) {
					//find the first sample exceeding the current threshold
					for(j=last+1; j<sortedidxsize; ++j) {
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
