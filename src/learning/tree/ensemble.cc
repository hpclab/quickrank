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
  for (unsigned int i = 0; i < size; ++i)
    delete arr[i].root;
  free(arr);
}

void Ensemble::set_capacity(const unsigned int n) {
  if (arr) {
    for (unsigned int i = 0; i < size; ++i)
      delete arr[i].root;
    free(arr);
  }
  arr = (wt*) malloc(sizeof(wt) * n), size = 0;
}

void Ensemble::push(RTNode *root, const float weight, const float maxlabel) {
  arr[size++] = wt(root, weight, maxlabel);
}

void Ensemble::pop() {
  delete arr[--size].root;
}

float Ensemble::eval(float * const * const features, unsigned int idx) const {
  float sum = 0.0f;
#pragma omp parallel for reduction(+:sum)
  for (unsigned int i = 0; i < size; ++i)
    sum += arr[i].root->eval(features, idx) * arr[i].weight;
  return sum;
}

// assumes vertical dataset
quickrank::Score Ensemble::score_instance(const quickrank::Feature* d,
                                          const unsigned int offset) const {
  double sum = 0.0f;
// #pragma omp parallel for reduction(+:sum)
  for (unsigned int i = 0; i < size; ++i)
    sum += arr[i].root->score_instance(d, offset) * arr[i].weight;
  return sum;
}

// TODO TO BE REMOVED
void Ensemble::write_outputtofile(FILE *f) {
  fprintf(f, "\n<ensemble>\n");
  for (unsigned int i = 0; i < size; ++i) {
    fprintf(f, "\t<tree id=\"%u\" weight=\"%.8f\">\n", i + 1, arr[i].weight);
    if (arr[i].root) {
      fprintf(f, "\t\t<split>\n");
      arr[i].root->write_outputtofile(f, 2);
      fprintf(f, "\t\t</split>\n");
    }
    fprintf(f, "\t</tree>\n");
  }
  fprintf(f, "</ensemble>\n");
}

std::ofstream& Ensemble::save_model_to_file(std::ofstream& os) const {
  auto old_precision = os.precision();
  os.setf(std::ios::floatfield, std::ios::fixed);
  os << "\t<ensemble>" << std::endl;
  for (unsigned int i = 0; i < size; ++i) {
    os << std::setprecision(3);
    os << "\t\t<tree id=\"" << i + 1 << "\" weight=\"" << arr[i].weight << "\">"
       << std::endl;
    if (arr[i].root) {
      os << "\t\t\t<split>" << std::endl;
      arr[i].root->save_model_to_file(os, 3);
      os << "\t\t\t</split>" << std::endl;
    }
    os << "\t\t</tree>" << std::endl;
  }
  os << "\t</ensemble>" << std::endl;
  os << std::setprecision(old_precision);
  return os;
}
