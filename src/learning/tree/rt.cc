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
#include "learning/tree/rt.h"

void DevianceMaxHeap::push_chidrenof(RTNode *parent) {
  push(parent->left->deviance, parent->left);
  push(parent->right->deviance, parent->right);
}
void DevianceMaxHeap::pop() {
  RTNode *node = top();
  delete[] node->sampleids, delete node->hist;
  node->sampleids = NULL, node->nsampleids = 0, node->hist = NULL;
  rt_maxheap::pop();
}

/// \todo TODO: memory management of regression tree is wrong!!!
RegressionTree::~RegressionTree() {
  if (root) {
    delete[] root->sampleids;
    root->sampleids = NULL, root->nsampleids = 0;
  }
  //if leaves[0] is the root, hist cannot be deallocated and sampleids has been already deallocated
  for (unsigned int i = 0; i < nleaves; ++i)
    if (leaves[i] != root) {
      delete[] leaves[i]->sampleids, delete leaves[i]->hist;
      leaves[i]->hist = NULL, leaves[i]->sampleids = NULL, leaves[i]->nsampleids =
          0;
    }
  free(leaves);
}

void RegressionTree::fit(RTNodeHistogram *hist) {
  DevianceMaxHeap heap(nrequiredleaves);
  unsigned int taken = 0;
  unsigned int nsampleids = training_dataset->num_instances();  //set->get_ndatapoints();
  unsigned int *sampleids = new unsigned int[nsampleids];
#pragma omp parallel for
  for (unsigned int i = 0; i < nsampleids; ++i)
    sampleids[i] = i;

  root = new RTNode(sampleids, hist);
  if (split(root, 1.0f, false))
    heap.push_chidrenof(root);
  while (heap.is_notempty()
      && (nrequiredleaves == 0 or taken + heap.get_size() < nrequiredleaves)) {
    //get node with highest deviance from heap
    RTNode *node = heap.top();
    // TODO: Cla missing check non leaf size or avoid putting them into the heap
    //try split current node
    if (split(node, 1.0f, false))
      heap.push_chidrenof(node);
    else
      ++taken;  //unsplitable (i.e. null variance, or after split variance is higher than before, or #samples<minlsd)
    //remove node from heap
    heap.pop();
  }
  //visit tree and save leaves in a leaves[] array
  unsigned int capacity = nrequiredleaves;
  leaves = capacity ? (RTNode**) malloc(sizeof(RTNode*) * capacity) : NULL, nleaves =
      0;
  root->save_leaves(leaves, nleaves, capacity);

  // TODO: (by cla) is memory of "unpopped" de-allocated?
}

double RegressionTree::update_output(double const *pseudoresponses) {
  double maxlabel = -DBL_MAX;
#pragma omp parallel for reduction(max:maxlabel)
  for (unsigned int i = 0; i < nleaves; ++i) {
    double psum = 0.0f;
    const unsigned int nsampleids = leaves[i]->nsampleids;
    const unsigned int *sampleids = leaves[i]->sampleids;
    for (unsigned int j = 0; j < nsampleids; ++j) {
      unsigned int k = sampleids[j];
      psum += pseudoresponses[k];
    }
    leaves[i]->avglabel = psum / nsampleids;

    if (leaves[i]->avglabel > maxlabel)
      maxlabel = leaves[i]->avglabel;
  }
  return maxlabel;
}

double RegressionTree::update_output(double const *pseudoresponses,
                                     double const *cachedweights) {
  double maxlabel = -DBL_MAX;
#pragma omp parallel for reduction(max:maxlabel)
  for (unsigned int i = 0; i < nleaves; ++i) {
    double s1 = 0.0;
    double s2 = 0.0;
    const unsigned int nsampleids = leaves[i]->nsampleids;
    const unsigned int *sampleids = leaves[i]->sampleids;
    for (unsigned int j = 0; j < nsampleids; ++j) {
      unsigned int k = sampleids[j];
      s1 += pseudoresponses[k];
      s2 += cachedweights[k];
      //					printf("## %d: %.15f \t %.15f \n", k, pseudoresponses[k], cachedweights[k]);
    }
    leaves[i]->avglabel = s2 >= DBL_EPSILON ? s1 / s2 : 0.0;

    //				printf("## Leaf with size: %d  ##  s1/s2: %.15f / %.15f = %.15f\n", nsampleids, s1, s2, leaves[i]->avglabel);

    if (leaves[i]->avglabel > maxlabel)
      maxlabel = leaves[i]->avglabel;
  }


  return maxlabel;
}

bool RegressionTree::split(RTNode *node, const float featuresamplingrate,
                           const bool require_devianceltparent) {
  //			printf("### Splitting a node of size: %d and deviance: %f\n", node->nsampleids, node->deviance);

  if (node->deviance > 0.0f) {
    // const double initvar = require_devianceltparent ? node->deviance : -1; //DBL_MAX;
    const double initvar = -1;  // minimum split score
    //get current nod hidtogram pointer
    RTNodeHistogram *h = node->hist;
    //featureidxs to be used for tree splitnodeting
    unsigned int nfeaturesamples = training_dataset->num_features();  //training_set->get_nfeatures();
    unsigned int *featuresamples = NULL;
    //need to make a sub-sampling
    if (featuresamplingrate < 1.0f) {
      featuresamples = new unsigned int[nfeaturesamples];
      for (unsigned int i = 0; i < nfeaturesamples; ++i)
        featuresamples[i] = i;
      //need to make a sub-sampling
      const unsigned int reduced_nfeaturesamples = floor(
          featuresamplingrate * nfeaturesamples);
      while (nfeaturesamples > reduced_nfeaturesamples && nfeaturesamples > 1) {
        const unsigned int i = rand() % nfeaturesamples;
        featuresamples[i] = featuresamples[--nfeaturesamples];
      }
    }
    // ---------------------------
    // find best split
    const int nth = omp_get_num_procs();
    double* thread_best_score = new double[nth];  // double thread_minvar[nth];
    unsigned int* thread_best_featureidx = new unsigned int[nth];  // unsigned int thread_best_featureidx[nth];
    unsigned int* thread_best_thresholdid = new unsigned int[nth];  // unsigned int thread_best_thresholdid[nth];
    for (int i = 0; i < nth; ++i)
      thread_best_score[i] = initvar, thread_best_featureidx[i] = uint_max, thread_best_thresholdid[i] =
          uint_max;

#pragma omp parallel for
    for (unsigned int i = 0; i < nfeaturesamples; ++i) {
      //get feature idx
      const unsigned int f = featuresamples ? featuresamples[i] : i;
      //get thread identification number
      const int ith = omp_get_thread_num();
      //define pointer shortcuts
      double *sumlabels = h->sumlbl[f];
      unsigned int *samplecount = h->count[f];
      //get last elements
      unsigned int threshold_size = h->thresholds_size[f];
      double s = sumlabels[threshold_size - 1];
      unsigned int c = samplecount[threshold_size - 1];

      //looking for the feature that minimizes sumvar
      for (unsigned int t = 0; t < threshold_size; ++t) {
        unsigned int lcount = samplecount[t];
        unsigned int rcount = c - lcount;
        if (lcount >= minls && rcount >= minls) {
          double lsum = sumlabels[t];
          double rsum = s - lsum;
          double score = lsum * lsum / (double) lcount
              + rsum * rsum / (double) rcount;
          if (score > thread_best_score[ith]) {
            thread_best_score[ith] = score;
            thread_best_featureidx[ith] = f;
            thread_best_thresholdid[ith] = t;
          }
        }
      }
    }
    //free feature samples
    delete[] featuresamples;
    //get best minvar among thread partial results
    double best_score = thread_best_score[0];
    unsigned int best_featureidx = thread_best_featureidx[0];
    unsigned int best_thresholdid = thread_best_thresholdid[0];
    for (int i = 1; i < nth; ++i)
      if (thread_best_score[i] > best_score)
        best_score = thread_best_score[i], best_featureidx =
            thread_best_featureidx[i], best_thresholdid =
            thread_best_thresholdid[i];
    // free some memory
    delete[] thread_best_score;
    delete[] featuresamples;
    delete[] thread_best_featureidx;
    delete[] thread_best_thresholdid;
    //if minvar is the same of initvalue then the node is unsplitable
    if (best_score == initvar)
      return false;

    //set some result values related to minvar
    const unsigned int last_thresholdidx = h->thresholds_size[best_featureidx]
        - 1;
    const float best_threshold =
        h->thresholds[best_featureidx][best_thresholdid];

    const unsigned int count = h->count[best_featureidx][last_thresholdidx];
    const unsigned int lcount = h->count[best_featureidx][best_thresholdid];
    const unsigned int rcount = count - lcount;

    //split samples between left and right child
    unsigned int *lsamples = new unsigned int[lcount], lsize = 0;
    unsigned int *rsamples = new unsigned int[rcount], rsize = 0;
    float const* features = training_dataset->at(0, best_featureidx);  //training_set->get_fvector(best_featureidx);
    for (unsigned int i = 0, nsampleids = node->nsampleids; i < nsampleids;
        ++i) {
      unsigned int k = node->sampleids[i];
      if (features[k] <= best_threshold)
        lsamples[lsize++] = k;
      else
        rsamples[rsize++] = k;
    }
    //create histograms for children
    RTNodeHistogram *lhist = new RTNodeHistogram(node->hist, lsamples, lsize,
                                                 training_labels);
    RTNodeHistogram *rhist = NULL;
    if (node == root)
      rhist = new RTNodeHistogram(node->hist, lhist);
    else {
      //save some new/delete by converting parent histogram into the right-child one
      node->hist->transform_intorightchild(lhist), rhist = node->hist;
      node->hist = NULL;
    }

    //update current node
    node->set_feature(
        best_featureidx,
        best_featureidx + 1/*training_set->get_featureid(best_featureidx)*/);
    node->threshold = best_threshold;

    //create children
    node->left = new RTNode(lsamples, lhist);
    node->right = new RTNode(rsamples, rhist);

    // rhist->quick_dump(128,10);
    // lhist->quick_dump(25,10);

    return true;
  }
  return false;
}
