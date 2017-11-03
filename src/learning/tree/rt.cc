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

#ifdef _OPENMP
#include <omp.h>
#include <random>
#include <chrono>
#include <algorithm>
#else
#include "utils/omp-stubs.h"
#endif

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
  for (size_t i = 0; i < nleaves; ++i)
    if (leaves[i] != root) {
      delete[] leaves[i]->sampleids, delete leaves[i]->hist;
      leaves[i]->hist = NULL, leaves[i]->sampleids = NULL,
          leaves[i]->nsampleids =
              0;
    }
  free(leaves);
}

void RegressionTree::fit(RTNodeHistogram *hist,
                         float subsample, float max_features) {
  DevianceMaxHeap heap(nrequiredleaves);
  size_t taken = 0;

  size_t nsampleids = training_dataset->num_instances();
  size_t *sampleids = new size_t[nsampleids];

  #pragma omp parallel for
  for (size_t i = 0; i < nsampleids; ++i)
    sampleids[i] = i;

//  //need to make a sub-sampling of query-doc pairs
//  if (subsample != 1.0f) {
//
//    featuresamples = new size_t[nfeaturesamples];
//    for (size_t i = 0; i < nfeaturesamples; ++i)
//      featuresamples[i] = i;
//    //need to make a sub-sampling
//    const size_t reduced_nfeaturesamples = (size_t) std::floor(
//        featuresamplingrate * nfeaturesamples);
//    while (nfeaturesamples > reduced_nfeaturesamples && nfeaturesamples > 1) {
//      const size_t i = rand() % nfeaturesamples;
//      featuresamples[i] = featuresamples[--nfeaturesamples];
//    }
//  }

  root = new RTNode(sampleids, hist);
  if (split(root, max_features, false))
    heap.push_chidrenof(root);
  while (heap.is_notempty()
      && (nrequiredleaves == 0 or taken + heap.get_size() < nrequiredleaves)) {
    //get node with highest deviance from heap
    RTNode *node = heap.top();
    // TODO: Cla missing check non leaf size or avoid putting them into the heap
    //try split current node
    if (split(node, max_features, false))
      heap.push_chidrenof(node);
    else
      ++taken;  //unsplitable (i.e. null variance, or after split variance is higher than before, or #samples<minlsd)
    //remove node from heap
    heap.pop();
  }
  //visit tree and save leaves in a leaves[] array
  size_t capacity = nrequiredleaves;
  leaves = capacity ? (RTNode **) malloc(sizeof(RTNode *) * capacity) : NULL,
      nleaves =
          0;
  root->save_leaves(leaves, nleaves, capacity);

  // TODO: (by cla) is memory of "unpopped" de-allocated?
}

double RegressionTree::update_output(double const *pseudoresponses) {
  double maxlabel = -DBL_MAX;
#pragma omp parallel for reduction(max:maxlabel)
  for (size_t i = 0; i < nleaves; ++i) {
    double psum = 0.0f;
    const size_t nsampleids = leaves[i]->nsampleids;
    const size_t *sampleids = leaves[i]->sampleids;
    for (size_t j = 0; j < nsampleids; ++j) {
      size_t k = sampleids[j];
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
  for (size_t i = 0; i < nleaves; ++i) {
    double s1 = 0.0;
    double s2 = 0.0;
    const size_t nsampleids = leaves[i]->nsampleids;
    const size_t *sampleids = leaves[i]->sampleids;
    for (size_t j = 0; j < nsampleids; ++j) {
      size_t k = sampleids[j];
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

bool RegressionTree::split(RTNode *node, const float max_features,
                           const bool require_devianceltparent) {

  if (node->deviance > 0.0f) {
    const double initvar = -1;  // minimum split score
    // get current node histogram pointer
    RTNodeHistogram *h = node->hist;

    // feature idx to be used for tree split node
    size_t nfeaturesamples = training_dataset->num_features();
    size_t *featuresamples = NULL; // NULL means it will use all the features

    //need to make a sub-sampling
    if (max_features != 1.0f) {

      size_t nfeatures = training_dataset->num_features();

      if (max_features > 1.0f) {
        // >1: Max feature is the number of features to use
        nfeaturesamples = (size_t) max_features;
      } else {
        // <1: Max feature is the fraction of features to use
        nfeaturesamples = std::min(nfeatures,
                                   (size_t) std::round(
                                       max_features * nfeatures));
      }

      featuresamples = new size_t[nfeatures];
      #pragma omp parallel for
      for (size_t i = 0; i < nfeatures; ++i)
        featuresamples[i] = i;

      // shuffle the sample idx
      auto seed = std::chrono::system_clock::now().time_since_epoch().count();
      auto rng = std::default_random_engine(seed);
      std::shuffle(&featuresamples[0], &featuresamples[nfeatures], rng);
    }

    // ---------------------------
    // find best split
    const int nth = omp_get_num_procs();
    double *thread_best_score = new double[nth];
    size_t *thread_best_featureidx = new size_t[nth];
    size_t *thread_best_thresholdid = new size_t[nth];
    for (int i = 0; i < nth; ++i) {
      thread_best_score[i] = initvar;
      thread_best_featureidx[i] = uint_max;
      thread_best_thresholdid[i] = uint_max;
    }

    #pragma omp parallel for
    for (size_t i = 0; i < nfeaturesamples; ++i) {
      //get feature idx
      const size_t f = featuresamples ? featuresamples[i] : i;
      //get thread identification number
      const int ith = omp_get_thread_num();
      //define pointer shortcuts
      double *sumlabels = h->sumlbl[f];
      size_t *samplecount = h->count[f];
      //get last elements
      size_t threshold_size = h->thresholds_size[f];
      double s = sumlabels[threshold_size - 1];
      size_t c = samplecount[threshold_size - 1];

      //looking for the feature that minimizes sumvar
      for (size_t t = 0; t < threshold_size; ++t) {
        size_t lcount = samplecount[t];
        size_t rcount = c - lcount;
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
    size_t best_featureidx = thread_best_featureidx[0];
    size_t best_thresholdid = thread_best_thresholdid[0];
    for (int i = 1; i < nth; ++i) {
      if (thread_best_score[i] > best_score) {
        best_score = thread_best_score[i];
        best_featureidx = thread_best_featureidx[i];
        best_thresholdid = thread_best_thresholdid[i];
      }
    }
    // free some memory
    delete[] thread_best_score;
    delete[] thread_best_featureidx;
    delete[] thread_best_thresholdid;
    //if minvar is the same of initvalue then the node is unsplitable
    if (best_score == initvar)
      return false;

    //set some result values related to minvar
    const size_t last_thresholdidx = h->thresholds_size[best_featureidx] - 1;
    const float best_threshold =
        h->thresholds[best_featureidx][best_thresholdid];

    const size_t count = h->count[best_featureidx][last_thresholdidx];
    const size_t lcount = h->count[best_featureidx][best_thresholdid];
    const size_t rcount = count - lcount;

    //split samples between left and right child
    size_t *lsamples = new size_t[lcount], lsize = 0;
    size_t *rsamples = new size_t[rcount], rsize = 0;
    float const *features = training_dataset->at(0, best_featureidx);
    for (size_t i = 0, nsampleids = node->nsampleids; i < nsampleids; ++i) {
      size_t k = node->sampleids[i];
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
        best_featureidx + 1);
    node->threshold = best_threshold;

    //create children
    node->left = new RTNode(lsamples, lhist);
    node->right = new RTNode(rsamples, rhist);

    return true;
  }
  return false;
}
