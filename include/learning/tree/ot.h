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
