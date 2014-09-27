#include "metric/mapscorer.h"

double MAPScorer::compute_score(const qlist &ql) {
  float ap = 0.0f;
  unsigned int count = 0;
  //compute score for relevant labels
  for(unsigned int i=0; i<ql.size; ++i)
    if(ql.labels[i]>0.0f)
      ap += (++count)/(i+1.0f);
  count = (ql.qid<nrelevantdocs && relevantdocs[ql.qid]>count) ? relevantdocs[ql.qid] : count;
  return count==0 ? 0.0f : ap/count;
}

fsymmatrix* MAPScorer::swap_change(const qlist &ql) {
  int* labels = new int [ql.size]; // int labels[ql.size];
  int* relcount = new int [ql.size]; // int relcount[ql.size];
  unsigned int count = 0;
  for(unsigned int i=0; i<ql.size; ++i) {
    if(ql.labels[i]>0.0f) //relevant if true
      labels[i] = 1,
      ++count;
    else
      labels[i] = 0;
    relcount[i] = count;
  }
  count = (ql.qid<nrelevantdocs && relevantdocs[ql.qid]>count) ? relevantdocs[ql.qid] : count;
  fsymmatrix *changes = new fsymmatrix(ql.size);
  //if(count==0)
  //	return changes; //all zeros
  if (count!=0) {
#pragma omp parallel for
    for(unsigned int i=0; i<ql.size-1; ++i)
      for(unsigned int j=i+1; j<ql.size; ++j)
        if(labels[i]!=labels[j]) {
          const int diff = labels[j]-labels[i];
          float change = ((relcount[i]+diff)*labels[j]-relcount[i]*labels[i])/(i+1.0f);
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
