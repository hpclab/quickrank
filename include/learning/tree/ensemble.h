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
