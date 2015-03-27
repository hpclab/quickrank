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
#ifndef QUICKRANK_LEARNING_TREE_OT_H_
#define QUICKRANK_LEARNING_TREE_OT_H_

#include <cfloat>
#include <cmath>

#include "learning/tree/rt.h"

class ObliviousRT : public RegressionTree {
 public:
  ObliviousRT(unsigned int nodes, quickrank::data::Dataset *dps, double *labels,
              unsigned int minls, unsigned int treedepth)
      : RegressionTree(nodes, dps, labels, minls),
        treedepth(treedepth) {
  }
  void fit(RTNodeHistogram *hist);

 protected:
  const unsigned int treedepth = 0;

 private:
  void fill(double **sumvar, const unsigned int nfeaturesamples,
            RTNodeHistogram const *hist);
  const double invalid = -DBL_MAX;
};

#endif
