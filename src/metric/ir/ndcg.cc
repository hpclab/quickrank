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
MetricScore Ndcg::compute_idcg(const quickrank::data::QueryResults* rl, const Score* scores) const {
  //make a copy of lables
  Label* copyoflabels = new Label[rl->num_results()];
  memcpy(copyoflabels, rl->labels(), sizeof(Label)*rl->num_results());
  //sort the copy
  std::sort(copyoflabels, copyoflabels+rl->num_results(), std::greater<int>());
  //compute dcg
  MetricScore dcg = compute_dcg(copyoflabels, rl->num_results(), cutoff());
  //free mem
  delete[] copyoflabels;
  //return dcg
  return dcg;
}



MetricScore Ndcg::evaluate_result_list(const ResultList& ql) const {
  if(ql.size==0) return 0.0;
  const unsigned int size = std::min(cutoff(),ql.size);
  const double idcg = Ndcg::compute_idcg(ql.labels, ql.size, size);
  return idcg > (MetricScore)0.0 ?
      (MetricScore) compute_dcg(ql.labels, ql.size, size)/idcg : (MetricScore)0.0;
}

MetricScore Ndcg::evaluate_result_list(const quickrank::data::QueryResults* rl, const Score* scores) const {
  if (rl->num_results()==0) return 0.0;
  const double idcg = Ndcg::compute_idcg(rl, scores);
  if (idcg>0)
    return Dcg::evaluate_result_list(rl,scores)/idcg;
  else
    return 0;
}


std::unique_ptr<Jacobian> Ndcg::get_jacobian(const ResultList &ql) const {
  const unsigned int size = std::min(cutoff(),ql.size);
  const double idcg = compute_idcg(ql.labels, ql.size, size);
  std::unique_ptr<Jacobian> changes =
      std::unique_ptr<Jacobian>( new Jacobian(ql.size) );
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

std::ostream& Ndcg::put(std::ostream& os) const {
  if (cutoff()!=Metric::NO_CUTOFF)
    return os << "NDCG@" << cutoff();
  else
    return os << "NDCG";
}

} // namespace ir
} // namespace metric
} // namespace qr
