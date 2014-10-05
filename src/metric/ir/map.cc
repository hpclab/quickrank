/*
 * map.cpp
 *
 *  Created on: Oct 3, 2014
 *      Author: claudio
 */
#include <cmath>
#include <algorithm>

#include "metric/ir/map.h"


namespace qr {
namespace metric {
namespace ir {


MetricScore Map::evaluate_result_list(const ResultList& ql) const {
  MetricScore ap = 0.0f;
  MetricScore count = 0.0f;
  for(unsigned int i=0; i<ql.size; ++i)
    if(ql.labels[i]>0.0f)
      ap += (++count)/(i+1.0f);
  return count>0 ? ap/count : 0.0;
}

std::unique_ptr<Jacobian> Map::get_jacobian(const ResultList &ql) const {
  int* labels = new int [ql.size]; // int labels[ql.size];
  int* relcount = new int [ql.size]; // int relcount[ql.size];
  MetricScore count = 0;
  for(unsigned int i=0; i<ql.size; ++i) {
    if(ql.labels[i]>0.0f) //relevant if true
      labels[i] = 1,
      ++count;
    else
      labels[i] = 0;
    relcount[i] = count;
  }
  // count = (ql.qid<nrelevantdocs && relevantdocs[ql.qid]>count) ? relevantdocs[ql.qid] : count;
  std::unique_ptr<Jacobian> changes =
      std::unique_ptr<Jacobian>( new Jacobian(ql.size) );
  if (count!=0) {
#pragma omp parallel for
    for(unsigned int i=0; i<ql.size-1; ++i)
      for(unsigned int j=i+1; j<ql.size; ++j)
        if(labels[i]!=labels[j]) {
          const int diff = labels[j]-labels[i];
          MetricScore change = ((relcount[i]+diff)*labels[j]-relcount[i]*labels[i])/(i+1.0f);
          for(unsigned int k=i+1; k<j; ++k)
            if(labels[k]>0) change += (relcount[k]+diff)/(k+1.0f);
          change += (-relcount[j]*diff)/(j+1.0f);
          changes->at(i,j) = change/count;
        }
  }
  delete [] labels;
  delete [] relcount;
  return changes;
}

void Map::print(std::ostream& os) const {
  if (cutoff()!=Metric::NO_CUTOFF)
    os << "MAP@" << cutoff();
  else
    os << "MAP";
}

} // namespace ir
} // namespace metric
} // namespace qr
