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
#include <iostream>
#include <iomanip>
#include <chrono>

#include <boost/container/list.hpp>
#include <boost/container/vector.hpp>

#include <boost/filesystem.hpp>

#include "io/svml.h"
#include "utils/strutils.h"

namespace quickrank {
namespace io {

// TODO: save info file or use mmap
// TODO: re-introduce multithreading
std::unique_ptr<data::Dataset> Svml::read_horizontal(
    const std::string &filename) {

  FILE *f = fopen(filename.c_str(), "r");
  if (!f) {
    std::cerr << "!!! Error while opening file " << filename << "."
              << std::endl;
    exit(EXIT_FAILURE);
  }

  file_size_ = boost::filesystem::file_size(filename);

  std::chrono::high_resolution_clock::time_point start_reading =
      std::chrono::high_resolution_clock::now();

  unsigned int maxfid = 0;

  // temporary copy of data
  boost::container::list<unsigned int> data_qids;
  boost::container::list<quickrank::Label> data_labels;
  boost::container::list<boost::container::vector<quickrank::Feature> > data_instances;

  while (not feof(f)) {
    //#pragma omp parallel for ordered reduction(max:maxfid) num_threads(4) schedule(static,1)
    //  for (int file_i=0; file_i<file_size_; file_i++) {
    //    if(feof(f)) {file_i = file_size_; continue; }
    ssize_t nread;
    size_t linelength = 0;
    char *line = NULL;
    //lines are read one-at-a-time by threads
    //#pragma omp ordered
    {
      nread = getline(&line, &linelength, f);
    }
    //if something is wrong with getline() or line is empty, skip to the next
    if (nread <= 0) {
      free(line);
      continue;
    }
    char *token = NULL, *pch = line;
    //skip initial spaces
    while (ISSPC(*pch) && *pch != '\0')
      ++pch;
    //skip comment line
    if (*pch == '#') {
      free(line);
      continue;
    }

    //read label (label is a mandatory field)
    if (ISEMPTY(token = read_token(pch)))
      exit(2);

    // read label and qid
    quickrank::Label relevance = atof(token);
    unsigned int qid = atou(read_token(pch), "qid:");

    // allocate feature vector and read instance
    boost::container::vector<quickrank::Feature> curr_instance(maxfid);

    //read a sequence of features, namely (fid,fval) pairs, then the ending description
    while (!ISEMPTY(token = read_token(pch, '#'))) {
      if (*token == '#') {
//#ifndef SKIP_DPDESCRIPTION
//        datapoint->set_description(strdup(++token));
//#endif
        *pch = '\0';
      } else {
        //read a feature (id,val) from token
        unsigned int fid = 0;
        float fval = 0.0f;
        if (sscanf(token, "%u:%f", &fid, &fval) != 2)
          exit(4);
        //add feature to the current dp
        if (fid > maxfid) {
          maxfid = fid;
          curr_instance.resize(maxfid);
        }
        curr_instance[fid - 1] = fval;
      }
    }

    // store partial data
    //#pragma omp ordered
    {
      data_qids.push_back(qid);
      data_labels.push_back(relevance);
      data_instances.push_back(boost::move(curr_instance));  // move should avoid copies
    }
    //free mem
    free(line);
  }
  //close input file
  fclose(f);

  std::chrono::high_resolution_clock::time_point start_processing =
      std::chrono::high_resolution_clock::now();

  // put partial data in final data structure
  data::Dataset* dataset = new data::Dataset(data_qids.size(), maxfid);
  auto i_q = data_qids.begin();
  auto i_l = data_labels.begin();
  auto i_x = data_instances.begin();
  while (i_q != data_qids.end()) {
    dataset->addInstance(*i_q, *i_l, boost::move(*i_x));
    i_q++;
    i_l++;
    i_x++;
  }

  std::chrono::high_resolution_clock::time_point end_processing =
      std::chrono::high_resolution_clock::now();

  reading_time_ = std::chrono::duration_cast<std::chrono::duration<double>>(
      start_processing - start_reading).count();

  processing_time_ = std::chrono::duration_cast<std::chrono::duration<double>>(
      end_processing - start_processing).count();

  return std::unique_ptr<data::Dataset>(dataset);
}

std::ostream& Svml::put(std::ostream& os) const {
  // num threads is not reported here.
  os << std::setprecision(2) << "#\t Reading time: " << reading_time_
     << " s. @ " << file_size_ / 1024 / 1024 / reading_time_ << " MB/s "
     << " (post-proc.: " << processing_time_ << " s.)" << std::endl;
  return os;
}

}  // namespace data
}  // namespace quickrank

/*
 LTR_VerticalDataset* Svml::read_vertical(const std::string &filename) const {

 LTR_VerticalDataset* v_dataset = new LTR_VerticalDataset();

 FILE *f = fopen(filename.c_str(), "r");
 if (f) {
 #ifdef SHOWTIMER
 double readingtimer = omp_get_wtime();
 #endif
 const int nth = omp_get_num_procs();
 unsigned int maxfid = INIT_NOFEATURES - 1;
 unsigned int linecounter = 0;
 unsigned int* th_ndps = new unsigned int[nth]();  // unsigned int th_ndps[nth];
 //for(int i=0; i<nth; ++i)
 //  th_ndps[i] = 0;
 BitArray* th_usedfid = new BitArray[nth];  // bitarray th_usedfid[nth];
 DataPointCollection coll;
 #pragma omp parallel num_threads(nth) shared(maxfid, linecounter)
 while (not feof(f)) {
 ssize_t nread;
 size_t linelength = 0;
 char *line = NULL;
 unsigned int nline = 0;
 //lines are read one-at-a-time by threads
 #pragma omp critical
 {
 nread = getline(&line, &linelength, f), nline = ++linecounter;
 }
 //if something is wrong with getline() or line is empty, skip to the next
 if (nread <= 0) {
 free(line);
 continue;
 }
 char *token = NULL, *pch = line;
 //skip initial spaces
 while (ISSPC(*pch) && *pch != '\0')
 ++pch;
 //skip comment line
 if (*pch == '#') {
 free(line);
 continue;
 }
 //each thread get its id to access th_usedfid[], th_ndps[]
 const int ith = omp_get_thread_num();
 //read label (label is a mandatory field)
 if (ISEMPTY(token = read_token(pch)))
 exit(2);
 //create a new dp for storing the max number of features seen till now
 DataPoint *datapoint = new DataPoint(atof(token), nline, maxfid + 1);
 //read qid (qid is a mandatory field)
 unsigned int qid = atou(read_token(pch), "qid:");
 //read a sequence of features, namely (fid,fval) pairs, then the ending description
 while (!ISEMPTY(token = read_token(pch, '#')))
 if (*token == '#') {
 #ifndef SKIP_DPDESCRIPTION
 datapoint->set_description(strdup(++token));
 #endif
 *pch = '\0';
 } else {
 //read a feature (id,val) from token
 unsigned int fid = 0;
 float fval = 0.0f;
 if (sscanf(token, "%u:%f", &fid, &fval) != 2)
 exit(4);
 //add feature to the current dp
 datapoint->ins_feature(fid, fval),
 //update used featureids
 th_usedfid[ith].set_up(fid),
 //update maxfid (it should be "atomically" managed but its consistency is not a problem)
 maxfid = fid > maxfid ? fid : maxfid;
 }
 //store current sample in trie
 #pragma omp critical
 {
 coll.insert(qid, datapoint);
 }
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
 for (int i = 1; i < nth; ++i)
 th_usedfid[0] |= th_usedfid[i], th_ndps[0] += th_ndps[i];
 //make an array of used features ids
 unsigned int nfeatureids = th_usedfid[0].get_upcounter();
 v_dataset->set_featureid_vector(th_usedfid[0].get_uparray(nfeatureids));
 //set counters
 v_dataset->set_ndatapoints(th_ndps[0]);
 v_dataset->set_nrankedlists(coll.get_nlists());
 v_dataset->set_nfeatures(v_dataset->get_featureid(nfeatureids - 1) + 1);
 DataPointList** dplists = coll.get_lists();
 // free some memory
 delete[] th_ndps;
 delete[] th_usedfid;
 //allocate memory
 #ifndef SKIP_DPDESCRIPTION
 descriptions = (char**)malloc(sizeof(char*)*ndps);
 #endif
 v_dataset->set_rloffsets_vector(
 (unsigned int*) malloc(
 sizeof(unsigned int) * (v_dataset->get_nrankedlists() + 1)));
 v_dataset->set_rlids_vector(
 (int*) malloc(sizeof(int) * v_dataset->get_nrankedlists()));
 v_dataset->set_labels_vector(
 (double*) malloc(sizeof(double) * v_dataset->get_ndatapoints()));
 unsigned int maxrlsize = 0;
 //compute 'rloffsets' values (i.e. prefixsum dplist sizes) and populate rlids
 for (unsigned int i = 0, sum = 0, rlsize = 0;
 i < v_dataset->get_nrankedlists(); ++i) {
 rlsize = dplists[i]->get_size();
 v_dataset->set_rlids(i, dplists[i]->get_qid());
 v_dataset->set_rloffsets(i, sum);
 maxrlsize = rlsize > maxrlsize ? rlsize : maxrlsize, sum += rlsize;
 }
 v_dataset->set_rloffsets(v_dataset->get_nrankedlists(),
 v_dataset->get_ndatapoints());
 //populate descriptions (if set), labels, and feature matrix (dp-major order)
 float **tmpfeatures = (float**) malloc(
 sizeof(float*) * v_dataset->get_ndatapoints());
 #pragma omp parallel for
 for (unsigned int i = 0; i < v_dataset->get_nrankedlists(); ++i) {
 #ifdef PRESERVE_DPFILEORDER
 dplists[i]->sort_bynline();
 #endif
 for (unsigned int j = v_dataset->get_rloffsets(i);
 j < v_dataset->get_rloffsets(i + 1); ++j) {
 DataPoint *front = dplists[i]->front();
 #ifndef SKIP_DPDESCRIPTION
 descriptions[j] = front->get_description(),
 #endif
 tmpfeatures[j] = front->get_resizedfeatures(v_dataset->get_nfeatures()), v_dataset
 ->set_label(j, front->get_label());
 dplists[i]->pop();
 }
 }
 //traspose current feature matrix to get a feature-major order matrix
 v_dataset->set_fmatrix(
 (float**) malloc(sizeof(float*) * v_dataset->get_nfeatures()));
 for (unsigned int i = 0; i < v_dataset->get_nfeatures(); ++i)
 v_dataset->set_fvector(
 i, (float*) malloc(sizeof(float) * v_dataset->get_ndatapoints()));
 transpose(v_dataset->get_fmatrix(), tmpfeatures,
 v_dataset->get_ndatapoints(), v_dataset->get_nfeatures());
 for (unsigned int i = 0; i < v_dataset->get_ndatapoints(); ++i)
 free(tmpfeatures[i]);
 free(tmpfeatures);
 //delete feature arrays related to skipped featureids and compact the feature matrix
 for (unsigned int i = 0, j = 0; i < nfeatureids; ++i, ++j) {
 while (j != v_dataset->get_featureid(i))
 free(v_dataset->get_fvector(j++));
 v_dataset->set_fvector(i, v_dataset->get_fvector(j));
 }
 v_dataset->set_nfeatures(nfeatureids);
 v_dataset->set_fmatrix(
 (float**) realloc(v_dataset->get_fmatrix(),
 sizeof(float*) * nfeatureids));
 //show statistics
 printf(
 "\tfile = %s\n\tno. of datapoints = %u\n\tno. of training queries = %u\n\tmax no. of datapoints in a training query = %u\n\tno. of features = %u\n",
 filename.c_str(), v_dataset->get_ndatapoints(),
 v_dataset->get_nrankedlists(), maxrlsize, v_dataset->get_nfeatures());
 #ifdef SHOWTIMER
 processingtimer = omp_get_wtime()-processingtimer;
 printf("\t\033[1melapsed reading time = %.3f seconds (%.0fMB/s, %d threads)\n\telapsed post-processing time = %.3f seconds\033[0m\n", readingtimer, filesize(filename.c_str())/readingtimer, nth, processingtimer);
 #endif
 //free mem from temporary data structures
 // TODO: (by cla) is each dplist deleted ?
 delete[] dplists;
 } else
 exit(5);

 return v_dataset;
 }
 */
