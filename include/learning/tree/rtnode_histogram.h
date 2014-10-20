#ifndef QUICKRANK_LEARNING_TREE_HISTOGRAM_H_
#define QUICKRANK_LEARNING_TREE_HISTOGRAM_H_

#include "data/ltrdata.h"
#include "data/dataset.h"


class RTNodeHistogram {
	public:
		float **thresholds = NULL; //[0..nfeatures-1]x[0..thresholds_size[i]-1]
		unsigned int const *thresholds_size = NULL;
		unsigned int **stmap = NULL; //[0..nfeatures-1]x[0..nthresholds-1]
		const unsigned int nfeatures = 0;
		double **sumlbl = NULL; //[0..nfeatures-1]x[0..nthresholds-1]
		double **sqsumlbl = NULL; //[0..nfeatures-1]x[0..nthresholds-1]
		unsigned int **count = NULL; //[0..nfeatures-1]x[0..nthresholds-1]
	public:
		RTNodeHistogram(float **thresholds, unsigned int const *thresholds_size, unsigned int nfeatures);

		RTNodeHistogram(RTNodeHistogram const* parent, unsigned int const* sampleids,
		          const unsigned int nsampleids, double const* labels);

		RTNodeHistogram(RTNodeHistogram const* parent, RTNodeHistogram const* left);

		~RTNodeHistogram();

		void update(double *labels, const unsigned int nlabels);

		void transform_intorightchild(RTNodeHistogram const* left);

		void quick_dump(unsigned int f, unsigned int num_t);
};

class RTRootHistogram : public RTNodeHistogram {
	public:
		RTRootHistogram(quickrank::data::Dataset *dps, double *labels, unsigned int **sortedidx,
		              unsigned int sortedidxsize, float **thresholds,
		              unsigned int const *thresholds_size);

		~RTRootHistogram();
};

#endif
