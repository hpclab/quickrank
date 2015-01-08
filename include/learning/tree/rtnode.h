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
#ifndef QUICKRANK_LEARNING_TREE_RTNODE_H_
#define QUICKRANK_LEARNING_TREE_RTNODE_H_

#include "learning/tree/rtnode_histogram.h"
#include "types.h"

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

  // number of internal nodes traversed
  static unsigned long long _internal_nodes_traversed;

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

  RTNode(unsigned int *new_sampleids, unsigned int new_nsampleids,
         double new_deviance, double sumlabel, RTNodeHistogram* new_hist) {
    sampleids = new_sampleids;
    nsampleids = new_nsampleids;
    deviance = new_deviance;
    hist = new_hist;
    avglabel = nsampleids ? sumlabel / nsampleids : 0.0;
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
    _internal_nodes_traversed += (featureidx == uint_max ? 0 : 1);
    return score;
  }

  static void clean_stats() {
    _internal_nodes_traversed = 0;
  }

  static unsigned long long internal_nodes_traversed() {
    return _internal_nodes_traversed;
  }

  void write_outputtofile(FILE *f, const int indentsize);
  std::ofstream& save_model_to_file(std::ofstream&, const int);
};

#endif
