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
#pragma omp parallel for reduction(+:sum)
  for (unsigned int i = 0; i < size; ++i)
    sum += arr[i].root->score_instance(d, offset) * arr[i].weight;
  return sum;
}

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
