#include "learning/dpset.h"

void DataPoint::ins_feature(const unsigned int fid, const float fval) {
  if(fid>=maxsize) {
    maxsize = 2*fid+1;
    features = (float*)realloc(features, sizeof(float)*maxsize);
  }
  for(unsigned int i=maxfid+1; i<fid; features[i++]=0.0f);
  maxfid = fid>maxfid ? fid : maxfid,
      features[fid] = fval;
}

float* DataPoint::get_resizedfeatures(const unsigned int size) {
  if(size>=maxsize and size!=0)
    features = (float*)realloc(features, sizeof(float)*size);
  for(unsigned int i=maxfid+1; i<size; features[i++]=0.0f);
  return features;
}



void DataPointCollection::insert(const unsigned int qid, DataPoint* x) {
  if(qid>=arrsize) {
    unsigned int newsize = 2*qid+1;
    arr = (DataPointList**)realloc(arr, sizeof(DataPointList*)*newsize);
    while(arrsize<newsize) arr[arrsize++] = NULL;
  }
  if(arr[qid]==NULL)
    arr[qid] = new DataPointList(qid),
    ++nlists;
  arr[qid]->push(x);
}

DataPointList** DataPointCollection::get_lists() {
  if(nlists==0)
    return NULL;
  DataPointList **ret = new DataPointList*[nlists];
  for(unsigned int i=0, j=0; i<arrsize; ++i)
    if(arr[i]!=NULL)
      ret[j++] = arr[i];
  return ret;
}

DataPointDataset::DataPointDataset(const char *filename) {
  FILE *f = fopen(filename, "r");
  if(f) {
#ifdef SHOWTIMER
    double readingtimer = omp_get_wtime();
#endif
    const int nth = omp_get_num_procs();
    unsigned int maxfid = INIT_NOFEATURES-1;
    unsigned int linecounter = 0;
    unsigned int* th_ndps = new unsigned int [nth] (); // unsigned int th_ndps[nth];
    //for(int i=0; i<nth; ++i)
    //	th_ndps[i] = 0;
    BitArray* th_usedfid = new BitArray[nth]; // bitarray th_usedfid[nth];
    DataPointCollection coll;
#pragma omp parallel num_threads(nth) shared(maxfid, linecounter)
    while(not feof(f)) {
      ssize_t nread;
      size_t linelength = 0;
      char *line = NULL;
      unsigned int nline = 0;
      //lines are read one-at-a-time by threads
#pragma omp critical
      { nread = getline(&line, &linelength, f), nline = ++linecounter; }
      //if something is wrong with getline() or line is empty, skip to the next
      if(nread<=0) { free(line); continue; }
      char *token = NULL, *pch = line;
      //skip initial spaces
      while(ISSPC(*pch) && *pch!='\0') ++pch;
      //skip comment line
      if(*pch=='#') { free(line); continue; }
      //each thread get its id to access th_usedfid[], th_ndps[]
      const int ith = omp_get_thread_num();
      //read label (label is a mandatory field)
      if(ISEMPTY(token=read_token(pch))) exit(2);
      //create a new dp for storing the max number of features seen till now
      DataPoint *datapoint = new DataPoint(atof(token), nline, maxfid+1);
      //read qid (qid is a mandatory field)
      unsigned int qid = atou(read_token(pch), "qid:");
      //read a sequence of features, namely (fid,fval) pairs, then the ending description
      while(!ISEMPTY(token=read_token(pch,'#')))
        if(*token=='#') {
#ifndef SKIP_DPDESCRIPTION
          datapoint->set_description(strdup(++token));
#endif
          *pch = '\0';
        } else {
          //read a feature (id,val) from token
          unsigned int fid = 0;
          float fval = 0.0f;
          if(sscanf(token, "%u:%f", &fid, &fval)!=2) exit(4);
          //add feature to the current dp
          datapoint->ins_feature(fid, fval),
              //update used featureids
              th_usedfid[ith].set_up(fid),
              //update maxfid (it should be "atomically" managed but its consistency is not a problem)
              maxfid = fid>maxfid ? fid : maxfid;
        }
      //store current sample in trie
#pragma omp critical
      { coll.insert(qid, datapoint); }
      //update thread dp counter
      ++th_ndps[ith],
      //free mem
      free(line);
    }
    //close input file
    fclose(f);
#ifdef SHOWTIMER
    readingtimer = omp_get_wtime()-readingtimer;
    double processingtimer = omp_get_wtime();
#endif
    //merge thread counters and compute the number of features
    for(int i=1; i<nth; ++i)
      th_usedfid[0] |= th_usedfid[i],
      th_ndps[0] += th_ndps[i];
    //make an array of used features ids
    unsigned int nfeatureids = th_usedfid[0].get_upcounter();
    usedfid = th_usedfid[0].get_uparray(nfeatureids);
    //set counters
    ndps = th_ndps[0],
        nrankedlists = coll.get_nlists(),
        nfeatures = usedfid[nfeatureids-1]+1;
    DataPointList** dplists = coll.get_lists();
    // free some memory
    delete [] th_ndps;
    delete [] th_usedfid;
    //allocate memory
#ifndef SKIP_DPDESCRIPTION
    descriptions = (char**)malloc(sizeof(char*)*ndps),
#endif
        rloffsets = (unsigned int*)malloc(sizeof(unsigned int)*(nrankedlists+1)),
        rlids = (int*)malloc(sizeof(int)*nrankedlists),
        labels = (double*)malloc(sizeof(double)*ndps);
    //compute 'rloffsets' values (i.e. prefixsum dplist sizes) and populate rlids
    for(unsigned int i=0, sum=0, rlsize=0; i<nrankedlists; ++i) {
      rlsize = dplists[i]->get_size(),
          rlids[i] = dplists[i]->get_qid(),
          rloffsets[i] = sum;
      maxrlsize = rlsize>maxrlsize ? rlsize : maxrlsize,
          sum += rlsize;
    }
    rloffsets[nrankedlists] = ndps;
    //populate descriptions (if set), labels, and feature matrix (dp-major order)
    float **tmpfeatures = (float**)malloc(sizeof(float*)*ndps);
#pragma omp parallel for
    for(unsigned int i=0; i<nrankedlists; ++i) {
#ifdef PRESERVE_DPFILEORDER
      dplists[i]->sort_bynline();
#endif
      for(unsigned int j=rloffsets[i]; j<rloffsets[i+1]; ++j) {
        DataPoint *front = dplists[i]->front();
#ifndef SKIP_DPDESCRIPTION
        descriptions[j] = front->get_description(),
#endif
            tmpfeatures[j] = front->get_resizedfeatures(nfeatures),
            labels[j] = front->get_label();
        dplists[i]->pop();
      }
    }
    //traspose current feature matrix to get a feature-major order matrix
    features = (float**)malloc(sizeof(float*)*nfeatures);
    for(unsigned int i=0; i<nfeatures; ++i)
      features[i] = (float*)malloc(sizeof(float)*ndps);
    transpose(features, tmpfeatures, ndps, nfeatures);
    for(unsigned int i=0; i<ndps; ++i)
      free(tmpfeatures[i]);
    free(tmpfeatures);
    //delete feature arrays related to skipped featureids and compact the feature matrix
    for(unsigned int i=0, j=0; i<nfeatureids; ++i, ++j) {
      while(j!=usedfid[i])
        free(features[j++]);
      features[i] = features[j];
    }
    nfeatures = nfeatureids,
        features = (float**)realloc(features, sizeof(float*)*nfeatureids);
    //show statistics
    printf("\tfile = %s\n\tno. of datapoints = %u\n\tno. of training queries = %u\n\tmax no. of datapoints in a training query = %u\n\tno. of features = %u\n", filename, ndps, nrankedlists, maxrlsize, nfeatures);
#ifdef SHOWTIMER
    processingtimer = omp_get_wtime()-processingtimer;
    printf("\t\033[1melapsed reading time = %.3f seconds (%.0fMB/s, %d threads)\n\telapsed post-processing time = %.3f seconds\033[0m\n", readingtimer, filesize(filename)/readingtimer, nth, processingtimer);
#endif
    //free mem from temporary data structures
    // TODO: (by cla) is each dplist deleted ?
    delete [] dplists;
  } else exit(5);
}
DataPointDataset::~DataPointDataset() {
  if(features) for(unsigned int i=0; i<nfeatures; ++i) free(features[i]);
#ifndef SKIP_DPDESCRIPTION
  if(descriptions) for(unsigned int i=0; i<ndps; ++i) free(descriptions[i]);
  free(descriptions),
#endif
      free(rloffsets),
      free(labels),
      free(features),
      free(rlids);
  delete [] usedfid;
}
