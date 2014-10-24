#include "learning/dpset.h"

void DataPoint::ins_feature(const unsigned int fid, const float fval) {
  if (fid >= maxsize) {
    maxsize = 2 * fid + 1;
    features = (float*) realloc(features, sizeof(float) * maxsize);
  }
  for (unsigned int i = maxfid + 1; i < fid; features[i++] = 0.0f)
    ;
  maxfid = fid > maxfid ? fid : maxfid, features[fid] = fval;
}

float* DataPoint::get_resizedfeatures(const unsigned int size) {
  if (size >= maxsize and size != 0)
    features = (float*) realloc(features, sizeof(float) * size);
  for (unsigned int i = maxfid + 1; i < size; features[i++] = 0.0f)
    ;
  return features;
}

void DataPointCollection::insert(const unsigned int qid, DataPoint* x) {
  if (qid >= arrsize) {
    unsigned int newsize = 2 * qid + 1;
    arr = (DataPointList**) realloc(arr, sizeof(DataPointList*) * newsize);
    while (arrsize < newsize)
      arr[arrsize++] = NULL;
  }
  if (arr[qid] == NULL)
    arr[qid] = new DataPointList(qid), ++nlists;
  arr[qid]->push(x);
}

DataPointList** DataPointCollection::get_lists() {
  if (nlists == 0)
    return NULL;
  DataPointList **ret = new DataPointList*[nlists];
  for (unsigned int i = 0, j = 0; i < arrsize; ++i)
    if (arr[i] != NULL)
      ret[j++] = arr[i];
  return ret;
}

