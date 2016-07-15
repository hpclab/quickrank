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
#include <fstream>
#include <iomanip>

#include "learning/tree/ensemble.h"

Ensemble::~Ensemble() {
  for (size_t i = 0; i < size; ++i)
    delete arr[i].root;
  free(arr);
}

void Ensemble::set_capacity(const size_t n) {
  if (arr) {
    for (size_t i = 0; i < size; ++i)
      delete arr[i].root;
    free(arr);
  }
  arr = (wt*) malloc(sizeof(wt) * n), size = 0;
}

void Ensemble::push(RTNode *root, const double weight, const float maxlabel) {
  arr[size++] = wt(root, weight, maxlabel);
}

void Ensemble::pop() {
  delete arr[--size].root;
}

// assumes vertical dataset
quickrank::Score Ensemble::score_instance(const quickrank::Feature* d,
                                          const size_t offset) const {
  double sum = 0.0f;
// #pragma omp parallel for reduction(+:sum)
  for (size_t i = 0; i < size; ++i)
    sum += arr[i].root->score_instance(d, offset) * arr[i].weight;
  return sum;
}

std::shared_ptr<std::vector<quickrank::Score>>
  Ensemble::partial_scores_instance(const quickrank::Feature *d,
                                    const size_t offset) const {
  std::vector<quickrank::Score> scores(size);
  for (unsigned int i = 0; i < size; ++i)
    scores[i] = arr[i].root->score_instance(d, offset) * arr[i].weight;
  return std::make_shared<std::vector<quickrank::Score>>(std::move(scores));
}

pugi::xml_node Ensemble::append_xml_model(pugi::xml_node parent,
                                          bool skip_useless_trees) const {

  pugi::xml_node ensemble = parent.append_child("ensemble");

  for (size_t i = 0; i < size; ++i) {
    // Use a small epsilon to check for null weights...
    if (skip_useless_trees && arr[i].weight < 0.0000001)
      continue;
    pugi::xml_node tree = ensemble.append_child("tree");
    tree.append_attribute("id") = i + 1;
    tree.append_attribute("weight") = arr[i].weight;
    if (arr[i].root) {
      arr[i].root->append_xml_model(tree);
    }
  }

  return ensemble;
}

bool Ensemble::update_ensemble_weights(
    std::shared_ptr<std::vector<double>> weights) {

  if (weights->size() != get_size()) {
    std::cerr << "# ## ERROR!! Ensemble size does not match size of the "
                     "weight vector in updating the weights" << std::endl;
    return false;
  }

  for (size_t i = 0; i < size; ++i)
    arr[i].weight = (*weights)[i];

  return true;
}

std::shared_ptr<std::vector<double>> Ensemble::get_weights() const {
  std::vector<double>* weights = new std::vector<double>(size);
  for (unsigned int i = 0; i < size; ++i)
    (*weights)[i] = arr[i].weight;
  return std::shared_ptr<std::vector<double>>(weights);
}