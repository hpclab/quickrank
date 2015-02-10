/*
 * QuickRank - A C++ suite of Learning to Rank algorithms
 * Webpage: http://quickrank.isti.cnr.it/
 * Contact: quickrank@isti.cnr.it
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
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
