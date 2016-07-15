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
#pragma once

#include "learning/tree/rt.h"
#include "types.h"
#include "pugixml/src/pugixml.hpp"

class Ensemble {

 public:
  virtual ~Ensemble();
  void set_capacity(const size_t n);
  void push(RTNode *root, const double weight, const float maxlabel);
  void pop();

  size_t get_size() const {
    return size;
  }

  bool is_notempty() const {
    return size > 0;
  }

  virtual quickrank::Score score_instance(const quickrank::Feature* d,
                                          const size_t offset = 1) const;

  virtual std::shared_ptr<std::vector<quickrank::Score>>
      partial_scores_instance(const quickrank::Feature *d,
                              const size_t offset = 1) const;

  pugi::xml_node append_xml_model(pugi::xml_node parent,
                                  bool skip_useless_trees = true) const;

  virtual bool update_ensemble_weights(
      std::shared_ptr<std::vector<double>> weights);

  virtual std::shared_ptr<std::vector<double>> get_weights() const;

 private:
  struct wt {
    wt(RTNode *root, double weight, float maxlabel)
        : root(root),
          weight(weight),
          maxlabel(maxlabel) {
    }
    RTNode *root = NULL;
    double weight = 0.0;
    float maxlabel = 0.0f;
  };
  size_t size = 0;
  wt *arr = NULL;
};
