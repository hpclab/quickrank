#include <iostream>
#include <boost/container/list.hpp>
#include <boost/container/vector.hpp>
#include <boost/timer/timer.hpp>

#include <boost/filesystem.hpp>

#include "io/svml.h"

namespace quickrank {
namespace io {



LTR_VerticalDataset* Svml::read_vertical(const char *filename) const {

  LTR_VerticalDataset* v_dataset = new LTR_VerticalDataset();

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
    //  th_ndps[i] = 0;
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
    v_dataset->set_featureid_vector( th_usedfid[0].get_uparray(nfeatureids) );
    //set counters
    v_dataset->set_ndatapoints(th_ndps[0]);
    v_dataset->set_nrankedlists(coll.get_nlists());
    v_dataset->set_nfeatures( v_dataset->get_featureid(nfeatureids-1)+1 );
    DataPointList** dplists = coll.get_lists();
    // free some memory
    delete [] th_ndps;
    delete [] th_usedfid;
    //allocate memory
#ifndef SKIP_DPDESCRIPTION
    descriptions = (char**)malloc(sizeof(char*)*ndps);
#endif
    v_dataset->set_rloffsets_vector( (unsigned int*)malloc(sizeof(unsigned int)*(v_dataset->get_nrankedlists()+1)) );
    v_dataset->set_rlids_vector( (int*)malloc(sizeof(int)*v_dataset->get_nrankedlists()) );
    v_dataset->set_labels_vector( (double*)malloc(sizeof(double)*v_dataset->get_ndatapoints()) );
    unsigned int maxrlsize = 0;
    //compute 'rloffsets' values (i.e. prefixsum dplist sizes) and populate rlids
    for(unsigned int i=0, sum=0, rlsize=0; i<v_dataset->get_nrankedlists(); ++i) {
      rlsize = dplists[i]->get_size();
      v_dataset->set_rlids(i,dplists[i]->get_qid());
      v_dataset->set_rloffsets(i,sum);
      maxrlsize = rlsize>maxrlsize ? rlsize : maxrlsize,
          sum += rlsize;
    }
    v_dataset->set_rloffsets(v_dataset->get_nrankedlists(),v_dataset->get_ndatapoints());
    //populate descriptions (if set), labels, and feature matrix (dp-major order)
    float **tmpfeatures = (float**)malloc(sizeof(float*)*v_dataset->get_ndatapoints());
#pragma omp parallel for
    for(unsigned int i=0; i<v_dataset->get_nrankedlists(); ++i) {
#ifdef PRESERVE_DPFILEORDER
      dplists[i]->sort_bynline();
#endif
      for(unsigned int j=v_dataset->get_rloffsets(i); j<v_dataset->get_rloffsets(i+1); ++j) {
        DataPoint *front = dplists[i]->front();
#ifndef SKIP_DPDESCRIPTION
        descriptions[j] = front->get_description(),
#endif
            tmpfeatures[j] = front->get_resizedfeatures(v_dataset->get_nfeatures()),
            v_dataset->set_label(j,front->get_label());
        dplists[i]->pop();
      }
    }
    //traspose current feature matrix to get a feature-major order matrix
    v_dataset->set_fmatrix( (float**)malloc(sizeof(float*)*v_dataset->get_nfeatures()) );
    for(unsigned int i=0; i<v_dataset->get_nfeatures(); ++i)
      v_dataset->set_fvector(i, (float*)malloc(sizeof(float)*v_dataset->get_ndatapoints()) );
    transpose(v_dataset->get_fmatrix(), tmpfeatures, v_dataset->get_ndatapoints(), v_dataset->get_nfeatures());
    for(unsigned int i=0; i<v_dataset->get_ndatapoints(); ++i)
      free(tmpfeatures[i]);
    free(tmpfeatures);
    //delete feature arrays related to skipped featureids and compact the feature matrix
    for(unsigned int i=0, j=0; i<nfeatureids; ++i, ++j) {
      while(j!=v_dataset->get_featureid(i))
        free(v_dataset->get_fvector(j++));
      v_dataset->set_fvector(i, v_dataset->get_fvector(j));
    }
    v_dataset->set_nfeatures( nfeatureids );
    v_dataset->set_fmatrix( (float**)realloc(v_dataset->get_fmatrix(), sizeof(float*)*nfeatureids) );
    //show statistics
    printf("\tfile = %s\n\tno. of datapoints = %u\n\tno. of training queries = %u\n\tmax no. of datapoints in a training query = %u\n\tno. of features = %u\n",
           filename, v_dataset->get_ndatapoints(), v_dataset->get_nrankedlists(), maxrlsize, v_dataset->get_nfeatures());
#ifdef SHOWTIMER
    processingtimer = omp_get_wtime()-processingtimer;
    printf("\t\033[1melapsed reading time = %.3f seconds (%.0fMB/s, %d threads)\n\telapsed post-processing time = %.3f seconds\033[0m\n", readingtimer, filesize(filename)/readingtimer, nth, processingtimer);
#endif
    //free mem from temporary data structures
    // TODO: (by cla) is each dplist deleted ?
    delete [] dplists;
  } else exit(5);

  return v_dataset;
}

// TODO: save info file or use mmap
// TODO: re-introduce multithreading
std::unique_ptr<data::Dataset> Svml::read_horizontal(const std::string &filename) const {

  FILE *f = fopen(filename.c_str(), "r");
  if(!f) {
    std::cerr << "!!! Error while opening file "<<filename<<"."<< std::endl;
    exit(EXIT_FAILURE);
  }

  boost::timer::cpu_timer reading_timer;

  unsigned int maxfid = 0;

  // temporary copy of data
  boost::container::list< unsigned int > data_qids;
  boost::container::list< quickrank::Label > data_labels;
  boost::container::list< boost::container::vector<quickrank::Feature> > data_instances;

  while(not feof(f)) {
    ssize_t nread;
    size_t linelength = 0;
    char *line = NULL;
    //lines are read one-at-a-time by threads
    nread = getline(&line, &linelength, f);
    //if something is wrong with getline() or line is empty, skip to the next
    if(nread<=0) { free(line); continue; }
    char *token = NULL, *pch = line;
    //skip initial spaces
    while(ISSPC(*pch) && *pch!='\0') ++pch;
    //skip comment line
    if(*pch=='#') { free(line); continue; }

    //read label (label is a mandatory field)
    if(ISEMPTY(token=read_token(pch))) exit(2);

    // read label and qid
    quickrank::Label relevance = atof(token);
    unsigned int qid = atou(read_token(pch), "qid:");

    // allocate feature vector and read instance
    boost::container::vector<quickrank::Feature> curr_instance(maxfid);

    //read a sequence of features, namely (fid,fval) pairs, then the ending description
    while(!ISEMPTY(token=read_token(pch,'#'))) {
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
        if (fid>maxfid) {
          maxfid = fid;
          curr_instance.resize(maxfid);
        }
        curr_instance[fid-1] = fval;
      }
    }

    // store partial data
    data_qids.push_back(qid);
    data_labels.push_back(relevance);
    data_instances.push_back( boost::move(curr_instance) ); // move should avoid copies

    //free mem
    free(line);
  }
  //close input file
  fclose(f);

  reading_timer.stop();

  // put partial data in final data structure
  boost::timer::cpu_timer processing_timer;
  data::Dataset* dataset = new data::Dataset(data_qids.size(), maxfid);
  auto i_q = data_qids.begin();
  auto i_l = data_labels.begin();
  auto i_x = data_instances.begin();
  while(i_q!=data_qids.end()) {
    dataset->addInstance( *i_q, *i_l, boost::move(*i_x) );
    i_q++;
    i_l++;
    i_x++;
  }

  processing_timer.stop();

  // num threads is not reported here.
  std::cout << "\t elapsed reading time = " << reading_timer.elapsed().wall/1000000000.0 << " sec.s ( "
      << boost::filesystem::file_size(filename)/1024/1024/(reading_timer.elapsed().wall/1000000000.0) << " MB/s) " << std::endl
      << "\t elapsed post-processing time = " << processing_timer.elapsed().wall/1000000000.0 << " sec.s." << std::endl;

  return std::unique_ptr<data::Dataset>(dataset);
}


} // namespace data
} // namespace quickrank
