#include "learning/tree/histogram.h"

RTNodeHistogram::RTNodeHistogram(float **thresholds, unsigned int const *thresholds_size, unsigned int nfeatures) : thresholds(thresholds), thresholds_size(thresholds_size), nfeatures(nfeatures) {
  sumlbl = new double*[nfeatures],
      sqsumlbl = new double*[nfeatures],
      count = new unsigned int*[nfeatures];
  for(unsigned int i=0; i<nfeatures; ++i) {
    const unsigned int threshold_size = thresholds_size[i];
    sumlbl[i] = new double[threshold_size]();
    sqsumlbl[i] = new double[threshold_size]();
    count[i] = new unsigned int[threshold_size]();
  }
}
RTNodeHistogram::RTNodeHistogram(RTNodeHistogram const* parent, unsigned int const* sampleids, const unsigned int nsampleids, double const* labels) : RTNodeHistogram(parent->thresholds, parent->thresholds_size, parent->nfeatures) {
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
RTNodeHistogram::RTNodeHistogram(RTNodeHistogram const* parent, RTNodeHistogram const* left) : RTNodeHistogram(parent->thresholds, parent->thresholds_size, parent->nfeatures) {
  stmap = parent->stmap;
#pragma omp parallel for
  for(unsigned int i=0; i<nfeatures; ++i) {
    const unsigned int nthresholds = thresholds_size[i];
    for(unsigned int t=0; t<nthresholds; ++t)
      sumlbl[i][t] = parent->sumlbl[i][t]-left->sumlbl[i][t],
      sqsumlbl[i][t] = parent->sqsumlbl[i][t]-left->sqsumlbl[i][t],
      count[i][t] = parent->count[i][t]-left->count[i][t];
  }
}
RTNodeHistogram::~RTNodeHistogram() {
  for(unsigned int i=0; i<nfeatures; ++i)
    delete [] sumlbl[i],
    delete [] sqsumlbl[i],
    delete [] count[i];
  delete [] sumlbl,
  delete [] sqsumlbl,
  delete [] count;
}
void RTNodeHistogram::update(double *labels, const unsigned int nlabels) {
#pragma omp parallel for
  for(unsigned int i=0; i<nfeatures; ++i)
    for(unsigned int t=0; t<thresholds_size[i]; ++t)
      sumlbl[i][t] = 0.0,
      sqsumlbl[i][t] = 0.0;
#pragma omp parallel for
  for(unsigned int i=0; i<nfeatures; ++i)
    for(unsigned int j=0; j<nlabels; ++j) {
      const unsigned int t = stmap[i][j];
      sumlbl[i][t] += labels[j],
          sqsumlbl[i][t] += labels[j]*labels[j];
      //count doesn't change, so no need to re-compute
    }
#pragma omp parallel for
  for(unsigned int i=0; i<nfeatures; ++i)
    for(unsigned int t=1; t<thresholds_size[i]; ++t)
      sumlbl[i][t] += sumlbl[i][t-1],
      sqsumlbl[i][t] += sqsumlbl[i][t-1];
}
void RTNodeHistogram::transform_intorightchild(RTNodeHistogram const* left) {
#pragma omp parallel for
  for(unsigned int i=0; i<nfeatures; ++i) {
    const unsigned int nthresholds = thresholds_size[i];
    for(unsigned int t=0; t<nthresholds; ++t)
      sumlbl[i][t] -= left->sumlbl[i][t],
      sqsumlbl[i][t] -= left->sqsumlbl[i][t],
      count[i][t] -= left->count[i][t];
  }
}
void RTNodeHistogram::quick_dump(unsigned int f, unsigned int num_t) {
  printf("### Hist fx %d :", f);
  for(unsigned int t=0; t<num_t && t<thresholds_size[f]; t++)
    printf(" %f", sumlbl[f][t]);
  printf("\n");
}

RTRootHistogram::RTRootHistogram(DataPointDataset *dps, double *labels, unsigned int **sortedidx, unsigned int sortedidxsize, float **thresholds, unsigned int const *thresholds_size) : RTNodeHistogram(thresholds, thresholds_size, dps->get_nfeatures()) {
  stmap = new unsigned int*[nfeatures];
#pragma omp parallel for
  for(unsigned int i=0; i<nfeatures; ++i) {
    stmap[i] = new unsigned int[sortedidxsize];
    unsigned int threshold_size = thresholds_size[i];
    float *features = dps->get_fvector(i);
    float *threshold = thresholds[i];
    double sum = 0.0;
    double sqsum = 0.0;
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
RTRootHistogram::~RTRootHistogram() {
  for(unsigned int i=0; i<nfeatures; ++i)
    delete [] stmap[i];
  delete [] stmap;
}
