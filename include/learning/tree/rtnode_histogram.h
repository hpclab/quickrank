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
#ifndef QUICKRANK_LEARNING_TREE_HISTOGRAM_H_
#define QUICKRANK_LEARNING_TREE_HISTOGRAM_H_

#include "data/dataset.h"

class RTNodeHistogram {
 public:
  float **thresholds = NULL;  //[0..nfeatures-1]x[0..thresholds_size[i]-1]
  unsigned int const *thresholds_size = NULL;
  unsigned int **stmap = NULL;  //[0..nfeatures-1]x[0..nthresholds-1]
  const unsigned int nfeatures = 0;
  double **sumlbl = NULL;  //[0..nfeatures-1]x[0..nthresholds-1]
  unsigned int **count = NULL;  //[0..nfeatures-1]x[0..nthresholds-1]
  double squares_sum_ = 0.0;
 public:
  RTNodeHistogram(float **thresholds, unsigned int const *thresholds_size,
                  unsigned int nfeatures);

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
  RTRootHistogram(quickrank::data::Dataset *dps,
                  unsigned int **sortedidx, unsigned int sortedidxsize,
                  float **thresholds, unsigned int const *thresholds_size);

  ~RTRootHistogram();
};

#endif
