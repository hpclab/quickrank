/*
 * QuickRank - A C++ suite of Learning to Rank algorithms
 * Webpage: http://quickrank.isti.cnr.it/
 * Contact: quickrank@isti.cnr.it
 *
 * Unless explicitly acquired and licensed from Licensor under another
 * license, the contents of this file are subject to the Reciprocal Public
 * License ("RPL") Version 1.5, or subsequent versions as allowed by the RPL,
 * and You may not copy or use this file in either source code or executable
 * form, except in compliance with the terms and conditions of the RPL.
 *
 * All software distributed under the RPL is provided strictly on an "AS
 * IS" basis, WITHOUT WARRANTY OF ANY KIND, EITHER EXPRESS OR IMPLIED, AND
 * LICENSOR HEREBY DISCLAIMS ALL SUCH WARRANTIES, INCLUDING WITHOUT
 * LIMITATION, ANY WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
 * PURPOSE, QUIET ENJOYMENT, OR NON-INFRINGEMENT. See the RPL for specific
 * language governing rights and limitations under the RPL.
 *
 * Contributor:
 *   HPC. Laboratory - ISTI - CNR - http://hpc.isti.cnr.it/
 */
#include "learning/tree/rtnode_histogram.h"

RTNodeHistogram::RTNodeHistogram(float **thresholds,
                                 unsigned int const *thresholds_size,
                                 unsigned int nfeatures)
    : thresholds(thresholds),
      thresholds_size(thresholds_size),
      nfeatures(nfeatures),
      squares_sum_(0.0) {
  sumlbl = new double*[nfeatures];
  count = new unsigned int*[nfeatures];
  for (unsigned int i = 0; i < nfeatures; ++i) {
    const unsigned int threshold_size = thresholds_size[i];
    sumlbl[i] = new double[threshold_size]();
    count[i] = new unsigned int[threshold_size]();
  }
}
RTNodeHistogram::RTNodeHistogram(RTNodeHistogram const* parent,
                                 unsigned int const* sampleids,
                                 const unsigned int nsampleids,
                                 double const* labels)
    : RTNodeHistogram(parent->thresholds, parent->thresholds_size,
                      parent->nfeatures) {
  stmap = parent->stmap;
#pragma omp parallel for
  for (unsigned int i = 0; i < nfeatures; ++i) {
    for (unsigned int j = 0; j < nsampleids; ++j) {
      const unsigned int k = sampleids[j];
      const unsigned int t = stmap[i][k];
      sumlbl[i][t] += labels[k];
      count[i][t]++;
    }
    for (unsigned int t = 1; t < thresholds_size[i]; ++t) {
      sumlbl[i][t] += sumlbl[i][t - 1];
      count[i][t] += count[i][t - 1];
    }
  }
  squares_sum_ = 0.0;
  for (unsigned int j = 0; j < nsampleids; ++j) {
    const unsigned int k = sampleids[j];
    squares_sum_ += labels[k] * labels[k];
  }
}

RTNodeHistogram::RTNodeHistogram(RTNodeHistogram const* parent,
                                 RTNodeHistogram const* left)
    : RTNodeHistogram(parent->thresholds, parent->thresholds_size,
                      parent->nfeatures) {
  stmap = parent->stmap;
#pragma omp parallel for
  for (unsigned int i = 0; i < nfeatures; ++i) {
    const unsigned int nthresholds = thresholds_size[i];
    for (unsigned int t = 0; t < nthresholds; ++t) {
      sumlbl[i][t] = parent->sumlbl[i][t] - left->sumlbl[i][t];
      count[i][t] = parent->count[i][t] - left->count[i][t];
    }
  }
  squares_sum_ = parent->squares_sum_ - left->squares_sum_;
}

RTNodeHistogram::~RTNodeHistogram() {
  for (unsigned int i = 0; i < nfeatures; ++i)
    delete[] sumlbl[i], delete[] count[i];
  delete[] sumlbl, delete[] count;
}

void RTNodeHistogram::update(double *labels, const unsigned int nlabels) {
#pragma omp parallel for
  for (unsigned int i = 0; i < nfeatures; ++i)
    for (unsigned int t = 0; t < thresholds_size[i]; ++t) {
      sumlbl[i][t] = 0.0;
    }
#pragma omp parallel for
  for (unsigned int i = 0; i < nfeatures; ++i)
    for (unsigned int j = 0; j < nlabels; ++j) {
      const unsigned int t = stmap[i][j];
      sumlbl[i][t] += labels[j];
      //count doesn't change, so no need to re-compute
    }
#pragma omp parallel for
  for (unsigned int i = 0; i < nfeatures; ++i)
    for (unsigned int t = 1; t < thresholds_size[i]; ++t) {
      sumlbl[i][t] += sumlbl[i][t - 1];
    }
  squares_sum_ = 0.0;
  for (unsigned int k = 0; k < nlabels; ++k) {
    squares_sum_ += labels[k] * labels[k];
  }
}

void RTNodeHistogram::transform_intorightchild(RTNodeHistogram const* left) {
  squares_sum_ = squares_sum_ - left->squares_sum_;
#pragma omp parallel for
  for (unsigned int i = 0; i < nfeatures; ++i) {
    const unsigned int nthresholds = thresholds_size[i];
    for (unsigned int t = 0; t < nthresholds; ++t) {
      sumlbl[i][t] -= left->sumlbl[i][t];
      count[i][t] -= left->count[i][t];
    }
  }
}
void RTNodeHistogram::quick_dump(unsigned int f, unsigned int num_t) {
  printf("### Hist fx %d :", f);
  for (unsigned int t = 0; t < num_t && t < thresholds_size[f]; t++)
    printf(" %f", sumlbl[f][t]);
  printf("\n");
}

RTRootHistogram::RTRootHistogram(quickrank::data::Dataset *dps,
                                 unsigned int **sortedidx,
                                 unsigned int sortedidxsize, float **thresholds,
                                 unsigned int const *thresholds_size)
    : RTNodeHistogram(thresholds, thresholds_size, dps->num_features()) {
  stmap = new unsigned int*[nfeatures];
#pragma omp parallel for
  for (unsigned int i = 0; i < nfeatures; ++i) {
    stmap[i] = new unsigned int[sortedidxsize];
    unsigned int threshold_size = thresholds_size[i];
    float *features = dps->at(0, i);
    float *threshold = thresholds[i];
    for (unsigned int last = -1, j, t = 0; t < threshold_size; ++t) {
      //find the first sample exceeding the current threshold
      for (j = last + 1; j < sortedidxsize; ++j) {
        unsigned int k = sortedidx[i][j];
        if (features[k] > threshold[t])
          break;
        stmap[i][k] = t;
      }
      last = j - 1;
      count[i][t] = j;
    }
  }
}

RTRootHistogram::~RTRootHistogram() {
  for (unsigned int i = 0; i < nfeatures; ++i)
    delete[] stmap[i];
  delete[] stmap;
}
