/*
 * dcg.cpp
 *
 *  Created on: Oct 3, 2014
 *      Author: claudio
 */
#include <cmath>
#include <algorithm>

#include "metric/ir/dcg.h"

#include "utils/qsort.h" // quick sort (for small input)

namespace qr {
namespace metric {
namespace ir {

double Dcg::compute_dcg(double const* labels, const unsigned int nlabels, const unsigned int k) const {
  unsigned int size = std::min(k,nlabels);
  double dcg = 0.0;
#pragma omp parallel for reduction(+:dcg)
  for(unsigned int i=0; i<size; ++i)
    dcg += (pow(2.0,labels[i])-1.0f)/log2(i+2.0f);
  return dcg;
}

MetricScore Dcg::evaluate_result_list(const ResultList& ql) const {
  if(ql.size==0) return 0.0;
  const unsigned int size = std::min(cutoff(),ql.size);
  return (MetricScore) Dcg::compute_dcg(ql.labels, ql.size, size);
}

std::unique_ptr<Jacobian> Dcg::get_jacobian(const ResultList &ql) const {
  const unsigned int size = std::min(cutoff(),ql.size);
  std::unique_ptr<Jacobian> changes = std::unique_ptr<Jacobian>( new Jacobian(ql.size) );
#pragma omp parallel for
  for(unsigned int i=0; i<size; ++i) {
    //get the pointer to the i-th line of matrix
    double *vchanges = changes->vectat(i, i+1);
    for(unsigned int j=i+1; j<ql.size; ++j) {
      *vchanges++ = ( 1.0f/log2((double)(i+2))-1.0f/log2((double)(j+2)) ) *
          ( pow(2.0,(double)ql.labels[i])-pow(2.0,(double)ql.labels[j]) );
    }
  }
  return changes;
}

void Dcg::print(std::ostream& os) const {
  if (cutoff()!=Metric::NO_CUTOFF)
    os << "DCG@" << cutoff();
  else
    os << "DCG";
}

} // namespace ir
} // namespace metric
} // namespace qr
