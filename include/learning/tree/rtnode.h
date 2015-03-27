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
#ifndef QUICKRANK_LEARNING_TREE_RTNODE_H_
#define QUICKRANK_LEARNING_TREE_RTNODE_H_

#include "learning/tree/rtnode_histogram.h"
#include "types.h"

#ifdef QUICKRANK_PERF_STATS
#include <atomic>
#endif

static const unsigned int uint_max = (unsigned int) -1;

class RTNode {

 public:
  unsigned int *sampleids = NULL;
  unsigned int nsampleids = 0;
  float threshold = 0.0f;
  double deviance = 0.0;
  double avglabel = 0.0;
  RTNode *left = NULL;
  RTNode *right = NULL;
  RTNodeHistogram *hist = NULL;

 private:
  unsigned int featureidx = uint_max;  //refer the index in the feature matrix
  unsigned int featureid = uint_max;  //refer to the id occurring in the dataset file

#ifdef QUICKRANK_PERF_STATS
  // number of internal nodes traversed
  static std::atomic<std::uint_fast64_t> _internal_nodes_traversed;
#endif

 public:
  // new leaf
  RTNode(double prediction) {
    avglabel = prediction;
    /*
     featureidx  = uint_max;
     featureid  = uint_max;
     sampleids = NULL;
     nsampleids = 0;
     deviance = -1;
     hist = NULL;
     left = NULL;
     right = NULL;
     */
  }

  RTNode(unsigned int *new_sampleids, unsigned int new_nsampleids, double prediction) {
    sampleids = new_sampleids;
    nsampleids = new_nsampleids;
    avglabel = prediction;
  }

  // new node
  RTNode(float new_threshold, unsigned int new_featureidx,
         unsigned int new_featureid, RTNode* new_left, RTNode* new_right) {
    threshold = new_threshold;
    featureidx = new_featureidx;
    featureid = new_featureid;
    left = new_left;
    right = new_right;
    /*
     sampleids = NULL;
     nsampleids = 0;
     deviance = -1;
     hist = NULL;
     avglabel = 0.0;
     */
  }

  RTNode(unsigned int *new_sampleids, RTNodeHistogram* new_hist) {
    hist = new_hist;
    sampleids = new_sampleids;
    nsampleids = hist->count[0][hist->thresholds_size[0] - 1];
    double sumlabel = hist->sumlbl[0][hist->thresholds_size[0] - 1];
    avglabel = nsampleids ? sumlabel / (double) nsampleids : 0.0;
    deviance = hist->squares_sum_
        - hist->sumlbl[0][hist->thresholds_size[0] - 1]
            * hist->sumlbl[0][hist->thresholds_size[0] - 1]
            / (double) hist->count[0][hist->thresholds_size[0] - 1];
  }

  ~RTNode() {
    if (left)
      delete left;
    if (right)
      delete right;
  }
  void set_feature(unsigned int fidx, unsigned int fid) {
    //if(fidx==uint_max or fid==uint_max) exit(7);
    featureidx = fidx, featureid = fid;
  }
  unsigned int get_feature_id() {
    return featureid;
  }
  unsigned int get_feature_idx() {
    return featureidx;
  }

  void save_leaves(RTNode **&leaves, unsigned int &nleaves,
                   unsigned int &capacity);

  bool is_leaf() const {
    return featureidx == uint_max;
  }
  double eval(float const* const * featurematrix,
              const unsigned int idx) const {
    return
        featureidx == uint_max ?
            avglabel :
            (featurematrix[featureidx][idx] <= threshold ?
                left->eval(featurematrix, idx) : right->eval(featurematrix, idx));
  }

  quickrank::Score score_instance(const quickrank::Feature* d,
                                  const unsigned int offset) const {
    /*if (featureidx == uint_max)
     std::cout << avglabel << std::endl;
     else
     std::cout << d[featureidx * offset] << "<=" << threshold << std::endl;
     */
    quickrank::Score score =
        featureidx == uint_max ?
            avglabel :
            (d[featureidx * offset] <= threshold ?
                left->score_instance(d, offset) :
                right->score_instance(d, offset));
#ifdef QUICKRANK_PERF_STATS
    if (featureidx != uint_max)
    _internal_nodes_traversed.fetch_add(1, std::memory_order_relaxed);
#endif
    return score;
  }

#ifdef QUICKRANK_PERF_STATS
  static void clean_stats() {
    _internal_nodes_traversed = 0;
  }

  static unsigned long long internal_nodes_traversed() {
    return _internal_nodes_traversed;
  }
#endif

  void write_outputtofile(FILE *f, const int indentsize);
  std::ofstream& save_model_to_file(std::ofstream&, const int);
};

#endif
