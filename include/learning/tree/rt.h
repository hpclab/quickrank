#ifndef QUICKRANK_LEARNING_TREE_RT_H_
#define QUICKRANK_LEARNING_TREE_RT_H_

#include <cfloat>
#include <cmath>
#include <cstring>

#ifdef _OPENMP
#include <omp.h>
#else
#include "utils/omp-stubs.h"
#endif

#include "utils/maxheap.h"
#include "data/dataset.h"
#include "learning/tree/rtnode.h"
#include "learning/tree/rtnode_histogram.h"

typedef MaxHeap<RTNode*> rt_maxheap;

class DevianceMaxHeap : public rt_maxheap {
 public:
  DevianceMaxHeap(unsigned int initsize)
      : rt_maxheap(initsize) {
  }
  void push_chidrenof(RTNode *parent);
  void pop();
};

class RegressionTree {
 protected:
  const unsigned int nrequiredleaves;  //0 for unlimited number of nodes (the size of the tree will then be controlled only by minls)
  const unsigned int minls;  //minls>0
  // LTR_VerticalDataset *training_set = NULL;
  quickrank::data::Dataset* training_dataset = NULL;
  double *training_labels = NULL;
  RTNode **leaves = NULL;
  unsigned int nleaves = 0;
  RTNode *root = NULL;
 public:
  RegressionTree(unsigned int nrequiredleaves, quickrank::data::Dataset *dps,
                 double *labels, unsigned int minls)
      : nrequiredleaves(nrequiredleaves),
        minls(minls),
        training_dataset(dps),
        training_labels(labels) {
  }
  ~RegressionTree();

  void fit(RTNodeHistogram *hist);

  double update_output(double const *pseudoresponses,
                       double const *cachedweights);

  double eval(float const* const * featurematrix,
              const unsigned int idx) const {
    return root->eval(featurematrix, idx);
  }

  RTNode *get_proot() const {
    return root;
  }

 private:
  //if require_devianceltparent is true the node is split if minvar is lt the current node deviance (require_devianceltparent=false in RankLib)
  bool split(RTNode *node, const float featuresamplingrate,
             const bool require_devianceltparent);

};

#endif
