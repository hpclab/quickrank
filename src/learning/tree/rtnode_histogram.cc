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
                                 size_t *thresholds_size,
                                 size_t nfeatures)
    : thresholds(thresholds),
      thresholds_size(thresholds_size),
      nfeatures(nfeatures),
      squares_sum_(0.0) {

  sumlbl = new double *[nfeatures];
  count = new size_t *[nfeatures];
  for (size_t i = 0; i < nfeatures; ++i) {
    const size_t threshold_size = thresholds_size[i];
    sumlbl[i] = new double[threshold_size]();
    count[i] = new size_t[threshold_size]();
  }
}

RTNodeHistogram::RTNodeHistogram(RTNodeHistogram const *parent,
                                 size_t const *sampleids,
                                 const size_t nsampleids,
                                 double const *labels)
    : RTNodeHistogram(parent->thresholds,
                      parent->thresholds_size,
                      parent->nfeatures) {

  stmap = parent->stmap;

  #pragma omp parallel for
  for (size_t f = 0; f < nfeatures; ++f) {
    for (size_t i = 0; i < nsampleids; ++i) {
      const size_t s = sampleids[i];
      const size_t t = stmap[f][s];
      sumlbl[f][t] += labels[s];
      count[f][t]++;
    }
    for (size_t t = 1; t < thresholds_size[f]; ++t) {
      sumlbl[f][t] += sumlbl[f][t - 1];
      count[f][t] += count[f][t - 1];
    }
  }

  squares_sum_ = 0.0;
  for (size_t i = 0; i < nsampleids; ++i) {
    const size_t s = sampleids[i];
    squares_sum_ += labels[s] * labels[s];
  }
}

RTNodeHistogram::RTNodeHistogram(RTNodeHistogram const *parent,
                                 RTNodeHistogram const *left)
    : RTNodeHistogram(parent->thresholds,
                      parent->thresholds_size,
                      parent->nfeatures) {
  stmap = parent->stmap;

  #pragma omp parallel for
  for (size_t f = 0; f < nfeatures; ++f) {
    for (size_t t = 0; t < thresholds_size[f]; ++t) {
      sumlbl[f][t] = parent->sumlbl[f][t] - left->sumlbl[f][t];
      count[f][t] = parent->count[f][t] - left->count[f][t];
    }
  }
  squares_sum_ = parent->squares_sum_ - left->squares_sum_;
}

RTNodeHistogram::RTNodeHistogram(const RTNodeHistogram& source)
    : nfeatures(source.nfeatures) {
  squares_sum_ = source.squares_sum_;

  thresholds_size = new size_t[nfeatures];
  for (unsigned int f=0; f<nfeatures; ++f) {
    thresholds_size[f] = source.thresholds_size[f];
  }

  thresholds = new float*[nfeatures];
  for (unsigned int f=0; f<nfeatures; ++f) {
    thresholds[f] = new float[thresholds_size[f]];
    for (unsigned int t=0; t<thresholds_size[f]; ++t) {
      thresholds[f][t] = source.thresholds[f][t];
    }
  }

  stmap = new size_t*[nfeatures];
  for (unsigned int f=0; f<nfeatures; ++f) {
    stmap[f] = new size_t[thresholds_size[f]];
    for (unsigned int t=0; t<thresholds_size[f]; ++t) {
      stmap[f][t] = source.stmap[f][t];
    }
  }

  sumlbl = new double*[nfeatures];
  for (unsigned int f=0; f<nfeatures; ++f) {
    sumlbl[f] = new double[thresholds_size[f]];
    for (unsigned int t=0; t<thresholds_size[f]; ++t) {
      sumlbl[f][t] = source.sumlbl[f][t];
    }
  }

  count = new size_t*[nfeatures];
  for (unsigned int f=0; f<nfeatures; ++f) {
    count[f] = new size_t[thresholds_size[f]];
    for (unsigned int t=0; t<thresholds_size[f]; ++t) {
      count[f][t] = source.count[f][t];
    }
  }
}

RTNodeHistogram::~RTNodeHistogram() {
  for (size_t i = 0; i < nfeatures; ++i) {
    delete[] sumlbl[i];
    delete[] count[i];

  }
  delete[] sumlbl;
  delete[] count;
}

void RTNodeHistogram::update(double *labels, const size_t nlabels) {

  #pragma omp parallel for
  for (size_t f = 0; f < nfeatures; ++f) {
    for (size_t t = 0; t < thresholds_size[f]; ++t) {
      sumlbl[f][t] = 0.0;
    }
  }

  #pragma omp parallel for
  for (size_t f = 0; f < nfeatures; ++f) {
    for (size_t i = 0; i < nlabels; ++i) {
      const size_t t = stmap[f][i];
      sumlbl[f][t] += labels[i];
      //count doesn't change, so no need to re-compute
    }
  }

  #pragma omp parallel for
  for (size_t f = 0; f < nfeatures; ++f) {
    for (size_t t = 1; t < thresholds_size[f]; ++t) {
      sumlbl[f][t] += sumlbl[f][t - 1];
    }
  }

  squares_sum_ = 0.0;
  for (size_t k = 0; k < nlabels; ++k) {
    squares_sum_ += labels[k] * labels[k];
  }
}

void RTNodeHistogram::update(double *labels,
                             const size_t nsampleids, const size_t *sampleids) {

  #pragma omp parallel for
  for (size_t f = 0; f < nfeatures; ++f) {
    for (size_t t = 0; t < thresholds_size[f]; ++t) {
      sumlbl[f][t] = 0.0;
      count[f][t] = 0;
    }
  }

  #pragma omp parallel for
  for (size_t f = 0; f < nfeatures; ++f) {
    for (size_t j = 0; j < nsampleids; ++j) {
      const size_t s = sampleids[j];
      const size_t t = stmap[f][s];
      sumlbl[f][t] += labels[s];
      count[f][t]++;
      //count change, so we need to re-compute it!!
    }

    for (size_t t = 1; t < thresholds_size[f]; ++t) {
      sumlbl[f][t] += sumlbl[f][t - 1];
      count[f][t] += count[f][t - 1];
    }
  }

  squares_sum_ = 0.0;
  for (size_t k = 0; k < nsampleids; ++k) {
    const size_t s = sampleids[k];
    squares_sum_ += labels[s] * labels[s];
  }
}

void RTNodeHistogram::transform_intorightchild(RTNodeHistogram const *left) {
  squares_sum_ = squares_sum_ - left->squares_sum_;

  #pragma omp parallel for
  for (size_t f = 0; f < nfeatures; ++f) {
    const size_t nthresholds = thresholds_size[f];
    for (size_t t = 0; t < nthresholds; ++t) {
      sumlbl[f][t] -= left->sumlbl[f][t];
      count[f][t] -= left->count[f][t];
    }
  }
}

void RTNodeHistogram::quick_dump(size_t f, size_t num_t) {
  printf("### Hist fx %zu :", f);
  for (size_t t = 0; t < num_t && t < thresholds_size[f]; t++)
    printf(" %f", sumlbl[f][t]);
  printf("\n");
}


RTRootHistogram::RTRootHistogram(quickrank::data::VerticalDataset *dataset,
                                 size_t **sortedidx, size_t sortedidxsize,
                                 float **thresholds, size_t *thresholds_size)
    : RTNodeHistogram(thresholds, thresholds_size, dataset->num_features()) {

  stmap = new size_t *[nfeatures];

  #pragma omp parallel for
  for (size_t f = 0; f < nfeatures; ++f) {
    stmap[f] = new size_t[sortedidxsize];
    size_t threshold_size = thresholds_size[f];
    float *features = dataset->at(0, f);
    float *threshold = thresholds[f];
    size_t last = -1, j;
    for (size_t t = 0; t < threshold_size; ++t) {
      //find the first sample exceeding the current threshold
      for (j = last + 1; j < sortedidxsize; ++j) {
        size_t k = sortedidx[f][j];
        if (features[k] > threshold[t])
          break;
        stmap[f][k] = t;
      }
      last = j - 1;
      count[f][t] = j;
    }
  }
}

RTRootHistogram::~RTRootHistogram() {
  for (size_t i = 0; i < nfeatures; ++i)
    delete[] stmap[i];
  delete[] stmap;
}
