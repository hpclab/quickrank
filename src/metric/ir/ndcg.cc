/*
 * ndcg.cpp
 *
 *  Created on: Oct 3, 2014
 *      Author: claudio
 */
#include <cmath>
#include <algorithm>

#include "metric/ir/ndcg.h"

#include "utils/qsort.h" // quick sort (for small input)

namespace qr {
namespace metric {
namespace ir {

/*! Compute the ideal Discounted Cumulative Gain (iDCG) for a list of labels.
 * @param labels input values.
 * @param nlabels number of input values.
 * @param k maximum number of entities that can be recommended.
 * @return iDCG@ \a k for computed on \a labels.
 */
double Ndcg::compute_idcg(double const* labels, const unsigned int nlabels, const unsigned int k) const {
  //make a copy of lables
  double *copyoflabels = new double[nlabels];
  memcpy(copyoflabels, labels, sizeof(double)*nlabels);
  //sort the copy
  double_qsort(copyoflabels, nlabels);
  //compute dcg
  double dcg = compute_dcg(copyoflabels, nlabels, k);
  //free mem
  delete[] copyoflabels;
  //return dcg
  return dcg;
}


// TODO: Yahoo! LTR returns 0.5 instead of 0.0
MetricScore Ndcg::evaluate_result_list(const qlist& ql) const {
  if(ql.size==0) return -1.0; //0.0;
  const unsigned int size = std::min(cutoff(),ql.size);
  const double idcg = Ndcg::compute_idcg(ql.labels, ql.size, size);
  return idcg > (MetricScore)0.0 ?
      (MetricScore) compute_dcg(ql.labels, ql.size, size)/idcg : (MetricScore)0.0;
}

Jacobian* Ndcg::get_jacobian(const qlist &ql) const {
  const unsigned int size = std::min(cutoff(),ql.size);
  const double idcg = compute_idcg(ql.labels, ql.size, size);
  Jacobian* changes = new Jacobian(ql.size);
  if(idcg>0.0) {
#pragma omp parallel for
    for(unsigned int i=0; i<size; ++i) {
      //get the pointer to the i-th line of matrix
      double *vchanges = changes->vectat(i, i+1);
      for(unsigned int j=i+1; j<ql.size; ++j) {
        *vchanges++ = ( 1.0f/log2((double)(i+2))-1.0f/log2((double)(j+2)) ) *
            ( pow(2.0,(double)ql.labels[i])-pow(2.0,(double)ql.labels[j]) ) / idcg;
      }
    }
  }
  return changes;
}

void Ndcg::print(std::ostream& os) const {
  if (cutoff()!=Metric::NO_CUTOFF)
    os << "NDCG@" << cutoff();
  else
    os << "NDCG";
}

} // namespace ir
} // namespace metric
} // namespace qr
