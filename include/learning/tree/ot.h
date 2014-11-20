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
