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

  double update_output(double const *pseudoresponses);

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
