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

Ensemble::Ensemble(Ensemble&& other) {
  size = other.size;
  capacity = other.capacity;
  arr = other.arr;
  // reset source object
  other.arr = nullptr;
  other.size = 0;
  other.capacity = 0;
}

Ensemble::~Ensemble() {
  if (arr)
    reset_state();
}

void Ensemble::reset_state() {
  if (arr) {
    for (size_t i = 0; i < size; ++i)
      delete arr[i].root;
    free(arr);
    arr = nullptr;
  }
  size = 0;
  capacity = 0;
}

Ensemble& Ensemble::operator=(Ensemble&& other) {
  if(this != &other) {
    if(arr)
      reset_state();

    size = other.size;
    capacity = other.capacity;
    arr = other.arr;
    // reset source object
    other.arr = nullptr;
    other.size = 0;
    other.capacity = 0;
  }

  return *this;
}

void Ensemble::set_capacity(const size_t n) {

  if (arr) {

    if (n < size) {
      // We need to call the destructor for the RTNode exceeding the new size
      for (size_t i = n; i < size; ++i)
        delete arr[i].root;
      size = n;
    }

    arr = (weighted_tree*) realloc(arr, sizeof(weighted_tree) * n);

  } else {
    arr = (weighted_tree*) malloc(sizeof(weighted_tree) * n);
    size = 0;
  }

  if (arr == nullptr) {
    free (arr);
    std::cerr << "Error (re)allocating ensemble memory";
    exit(1);
  }

  capacity = n;
}

void Ensemble::push(RTNode *root, const double weight, const float maxlabel) {
  if (size >= capacity) {
    std::cerr << "Error adding a new tree into the ensemble, capacity reached!";
    exit(1);
  }

  arr[size++] = weighted_tree(root, weight, maxlabel);
}

void Ensemble::pop() {
  delete arr[--size].root;
}

// assumes vertical dataset
quickrank::Score Ensemble::score_instance(const quickrank::Feature *d,
                                          const size_t offset) const {
  double sum = 0.0f;
// #pragma omp parallel for reduction(+:sum)
  for (size_t i = 0; i < size; ++i)
    sum += arr[i].root->score_instance(d, offset) * arr[i].weight;
  return sum;
}

std::shared_ptr<std::vector<quickrank::Score>>
Ensemble::partial_scores_instance(const quickrank::Feature *d,
                                  bool ignore_weights,
                                  const size_t offset) const {
  std::vector<quickrank::Score> scores(size);
  for (unsigned int i = 0; i < size; ++i) {
    scores[i] = arr[i].root->score_instance(d, offset);
    if (!ignore_weights)
      scores[i] *= arr[i].weight;
  }
  return std::make_shared<std::vector<quickrank::Score>>(std::move(scores));
}

pugi::xml_node Ensemble::append_xml_model(pugi::xml_node parent) const {

  pugi::xml_node ensemble = parent.append_child("ensemble");

  for (size_t i = 0; i < size; ++i) {
    pugi::xml_node tree = ensemble.append_child("tree");
    tree.append_attribute("id") = i + 1;
    tree.append_attribute("weight") = arr[i].weight;
    if (arr[i].root) {
      arr[i].root->append_xml_model(tree);
    }
  }

  return ensemble;
}

bool Ensemble::filter_out_zero_weighted_trees() {

  size_t idx_curr = 0;
  for (size_t i = 0; i < size; ++i) {
    if (arr[i].weight == 0) {
      // Remove 0-weight tree
      delete arr[i].root;
    } else {
      // Check if we need to move back the tree in the array of root trees
      if (idx_curr < i)
        arr[idx_curr] = arr[i];
      arr[idx_curr].weight = arr[i].weight;
      ++idx_curr;
    }
  }

  // Set the new size to the last element index (+1 because it is a size)
  size = idx_curr;

  return true;
}

bool Ensemble::update_ensemble_weights(
    std::vector<double>& weights, bool remove) {

  if (weights.size() != size) {
    std::cerr << "# ## ERROR!! Ensemble size does not match size of the "
        "weight vector in updating the weights" << std::endl;
    std::exit(-1);
    return false;
  }

  for (size_t i = 0; i < size; ++i)
    arr[i].weight = weights[i];

  if (remove)
    return filter_out_zero_weighted_trees();

  return true;
}

bool Ensemble::update_ensemble_weights(std::vector<double>& weights) {
  return update_ensemble_weights(weights, true);
}

std::vector<double> Ensemble::get_weights() const {
  std::vector<double> weights(size);
  for (unsigned int i = 0; i < size; ++i)
    weights[i] = arr[i].weight;
  return weights;
}