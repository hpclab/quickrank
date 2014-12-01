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
#ifndef QUICKRANK_LEARNING_TREE_ENSEMBLE_H_
#define QUICKRANK_LEARNING_TREE_ENSEMBLE_H_

#include "learning/tree/rt.h"
#include "types.h"

class Ensemble {

 public:
  virtual ~Ensemble();
  void set_capacity(const unsigned int n);
  void push(RTNode *root, const float weight, const float maxlabel);
  void pop();

  unsigned int get_size() const {
    return size;
  }
  bool is_notempty() const {
    return size > 0;
  }

  float eval(float * const * const features, unsigned int idx) const;

  // assumes vertical dataset
  virtual quickrank::Score score_instance(const quickrank::Feature* d,
                                          const unsigned int offset = 1) const;

  void write_outputtofile(FILE *f);
  std::ofstream& save_model_to_file(std::ofstream& os) const;

 private:
  struct wt {
    wt(RTNode *root, float weight, float maxlabel)
        : root(root),
          weight(weight),
          maxlabel(maxlabel) {
    }
    RTNode *root = NULL;
    float weight = 0.0f;
    float maxlabel = 0.0f;
  };
  unsigned int size = 0;
  wt *arr = NULL;
};

#endif
