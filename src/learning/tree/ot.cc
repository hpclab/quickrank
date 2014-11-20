#include "learning/tree/ot.h"

#define POWTWO(e) (1<<(e))

void ObliviousRT::fit(RTNodeHistogram *hist) {
  //by default get all sampleids in the training set
  unsigned int nsampleids = training_dataset->num_instances();
  unsigned int *sampleids = new unsigned int[nsampleids];
#pragma omp parallel for
  for (unsigned int i = 0; i < nsampleids; ++i)
    sampleids[i] = i;
  //featureidxs to be used for "tree"
  unsigned int nfeaturesamples = training_dataset->num_features();
  //histarray and nodearray store histograms and treenodes used in the entire procedure (i.e. the entire tree)
  RTNode **nodearray = new RTNode*[POWTWO(treedepth + 1)]();  //initialized NULL
  //init tree root
  nodearray[0] = root = new RTNode(sampleids, nsampleids, DBL_MAX, 0.0, hist);
  //allocate a matrix for each (feature,threshold)
  double **sumvar = new double*[nfeaturesamples];
  for (unsigned int i = 0; i < nfeaturesamples; ++i)
    sumvar[i] = new double[hist->thresholds_size[i]];
  //tree computation
  for (unsigned int depth = 0; depth < treedepth; ++depth) {
    const unsigned int lbegin = POWTWO(depth) - 1;  //index of first histogram belonging to the current level
    const unsigned int lend = POWTWO(depth+1) - 1;  //index of first histogram belonging to the next level
    //init matrix to zero
#pragma omp parallel for
    for (unsigned int i = 0; i < nfeaturesamples; ++i) {
      const unsigned int thresholds_size = hist->thresholds_size[i];
      for (unsigned int j = 0; j < thresholds_size; ++j)
        sumvar[i][j] = 0.0;
    }
    //for each histogram on the current depth (i.e. fringe) add variance of each (feature,threshold) in sumvar matrix
    for (unsigned int i = lbegin; i < lend; ++i)
      fill(sumvar, nfeaturesamples, nodearray[i]->hist);
    //find best split in the matrix
    const int nth = omp_get_num_procs();
    double* thread_minvar = new double[nth];  // double thread_minvar[nth];
    unsigned int* thread_best_featureidx = new unsigned int[nth];  // unsigned int thread_best_featureidx[nth];
    unsigned int* thread_best_thresholdid = new unsigned int[nth];  // unsigned int thread_best_thresholdid[nth];
    for (int i = 0; i < nth; ++i)
      thread_minvar[i] = DBL_MAX, thread_best_featureidx[i] = uint_max, thread_best_thresholdid[i] =
          uint_max;
#pragma omp parallel for
    for (unsigned int f = 0; f < nfeaturesamples; ++f) {
      const int ith = omp_get_thread_num();
      const unsigned int threshold_size = hist->thresholds_size[f];
      for (unsigned int t = 0; t < threshold_size; ++t)
        if (sumvar[f][t] != invalid && sumvar[f][t] < thread_minvar[ith])
          thread_minvar[ith] = sumvar[f][t], thread_best_featureidx[ith] = f, thread_best_thresholdid[ith] =
              t;
    }
    double minvar = thread_minvar[0];
    unsigned int best_featureidx = thread_best_featureidx[0];
    unsigned int best_thresholdid = thread_best_thresholdid[0];
    for (int i = 1; i < nth; ++i)
      if (thread_minvar[i] < minvar)
        minvar = thread_minvar[i], best_featureidx = thread_best_featureidx[i], best_thresholdid =
            thread_best_thresholdid[i];
    delete[] thread_minvar;
    delete[] thread_best_featureidx;
    delete[] thread_best_thresholdid;
    if (minvar == invalid || minvar == DBL_MAX)
      break;  //node is unsplittable
    //init next depth
#pragma omp parallel for
    for (unsigned int i = lbegin; i < lend; ++i) {
      RTNode *node = nodearray[i];
      //calculate some values related to best_featureidx and best_thresholdid
      const unsigned int last_thresholdid =
          node->hist->thresholds_size[best_featureidx] - 1;
      const unsigned int lcount =
          node->hist->count[best_featureidx][best_thresholdid];
      const unsigned int rcount =
          node->hist->count[best_featureidx][last_thresholdid] - lcount;
      const double lsum = node->hist->sumlbl[best_featureidx][best_thresholdid];
      const double lsqsum =
          node->hist->sqsumlbl[best_featureidx][best_thresholdid];
      const double best_lvar = fabs(lsqsum - lsum * lsum / lcount);
      const double rsum = node->hist->sumlbl[best_featureidx][last_thresholdid]
                                                              - lsum;
      const double rsqsum =
          node->hist->sqsumlbl[best_featureidx][last_thresholdid] - lsqsum;
      const double best_rvar = fabs(rsqsum - rsum * rsum / rcount);
      const float best_threshold =
          node->hist->thresholds[best_featureidx][best_thresholdid];
      //split samples between left and right child
      unsigned int *lsamples = new unsigned int[lcount], lsize = 0;
      unsigned int *rsamples = new unsigned int[rcount], rsize = 0;
      float const* features = training_dataset->at(0, best_featureidx);  //training_set->get_fvector(best_featureidx);
      for (unsigned int j = 0, nsampleids = node->nsampleids; j < nsampleids;
          ++j) {
        const unsigned int k = node->sampleids[j];
        if (features[k] <= best_threshold)
          lsamples[lsize++] = k;
        else
          rsamples[rsize++] = k;
      }
      //create new histograms (except for the last level when nodes are leaves)
      RTNodeHistogram *lhist = NULL;
      RTNodeHistogram *rhist = NULL;
      if (depth != treedepth - 1) {
        lhist = new RTNodeHistogram(node->hist, lsamples, lsize,
                                    training_labels);
        if (node == root)
          rhist = new RTNodeHistogram(node->hist, lhist);
        else {
          //save some new/delete by converting parent histogram into the right-child one
          node->hist->transform_intorightchild(lhist), rhist = node->hist;
          node->hist = NULL;
        }
      }
      //update current node
      node->left = nodearray[2 * i + 1] =  new RTNode( lsamples, lsize, best_lvar, lsum, lhist);
      node->right =  nodearray[2 * i + 2] = new RTNode(rsamples, rsize, best_rvar, rsum, rhist);
      node->set_feature(best_featureidx, best_featureidx+1 /*training_set->get_featureid(best_featureidx)*/);
      node->threshold = best_threshold;
      node->deviance = minvar;
      //free mem
      if (depth) {
        delete node->hist, delete[] node->sampleids;
        node->hist = NULL, node->sampleids = NULL, node->nsampleids = 0;
      }
    }
  }
  //visit tree and save leaves in a leaves[] array
  unsigned int capacity = nrequiredleaves;
  leaves = capacity ? (RTNode**) malloc(sizeof(RTNode*) * capacity) : NULL, nleaves =
      0;
  root->save_leaves(leaves, nleaves, capacity);
  //free mem allocated for sumvar[][]
  for (unsigned int i = 0; i < nfeaturesamples; ++i)
    delete[] sumvar[i];
  delete[] sumvar,
  //delete temp data
  delete[] nodearray;
}

void ObliviousRT::fill(double **sumvar, const unsigned int nfeaturesamples,
                       RTNodeHistogram const *hist) {
#pragma omp parallel for
  for (unsigned int f = 0; f < nfeaturesamples; ++f) {
    //define pointer shortcuts
    double *sumlabels = hist->sumlbl[f];
    double *sqsumlabels = hist->sqsumlbl[f];
    unsigned int *samplecount = hist->count[f];
    //get last elements
    unsigned int threshold_size = hist->thresholds_size[f];
    double s = sumlabels[threshold_size - 1];
    double sq = sqsumlabels[threshold_size - 1];
    unsigned int c = samplecount[threshold_size - 1];
    //looking for the feature that minimizes sum of lvar+rvar
    for (unsigned int t = 0; t < threshold_size; ++t)
      if (sumvar[f][t] != invalid) {
        unsigned int lcount = samplecount[t];
        unsigned int rcount = c - lcount;
        if (lcount >= minls && rcount >= minls) {
          double lsum = sumlabels[t];
          double lsqsum = sqsumlabels[t];
          double rsum = s - lsum;
          double rsqsum = sq - lsqsum;
          sumvar[f][t] += fabs(lsqsum - lsum * lsum / lcount)
                  + fabs(rsqsum - rsum * rsum / rcount);
        } else
          sumvar[f][t] = invalid;
      }
  }
}

#undef POWTWO
