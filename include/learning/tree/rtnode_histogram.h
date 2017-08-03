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
#pragma once

#include "data/vertical_dataset.h"

class RTNodeHistogram {
 public:
  float **thresholds = NULL;  //[0..nfeatures-1]x[0..thresholds_size[i]-1]
  size_t *thresholds_size = NULL;
  size_t **stmap = NULL;  //[0..nfeatures-1]x[0..nthresholds-1]
  const size_t nfeatures = 0;
  double **sumlbl = NULL;  //[0..nfeatures-1]x[0..nthresholds-1]
  size_t **count = NULL;  //[0..nfeatures-1]x[0..nthresholds-1]
  double squares_sum_ = 0.0;

  RTNodeHistogram(float **thresholds,
                  size_t *thresholds_size,
                  size_t nfeatures);

  RTNodeHistogram(RTNodeHistogram const *parent,
                  size_t const *sampleids,
                  const size_t nsampleids,
                  double const *labels);

  RTNodeHistogram(RTNodeHistogram const *parent,
                  RTNodeHistogram const *left);

  RTNodeHistogram(const RTNodeHistogram& source);

  ~RTNodeHistogram();

  void update(double *labels,
              const size_t nlabels);

  void transform_intorightchild(RTNodeHistogram const *left);

  void quick_dump(size_t f, size_t num_t);
};

class RTRootHistogram: public RTNodeHistogram {
 public:
  RTRootHistogram(quickrank::data::VerticalDataset *dps,
                  size_t **sortedidx,
                  size_t sortedidxsize,
                  float **thresholds,
                  size_t *thresholds_size);

  ~RTRootHistogram();
};
