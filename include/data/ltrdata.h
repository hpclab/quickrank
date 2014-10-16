#ifndef QUICKRANK_DATA_LTRDATA_H_
#define QUICKRANK_DATA_LTRDATA_H_


#include "learning/dpset.h"

class LTR_VerticalDataset {
  public:

    LTR_VerticalDataset(){};

    virtual ~LTR_VerticalDataset();

    unsigned int get_nfeatures() const { return nfeatures; }
    void set_nfeatures(const unsigned int n) { nfeatures=n; }

    unsigned int get_ndatapoints() const { return ndps; }
    void set_ndatapoints(const unsigned int n) { ndps = n; }

    unsigned int get_nrankedlists() const { return nrankedlists; }
    void set_nrankedlists(const unsigned int n) { nrankedlists = n; }

    ResultList get_qlist(unsigned int i) const {
      return ResultList(rloffsets[i+1]-rloffsets[i], labels+rloffsets[i], rlids[i]);
    }

    // used to find uniq values, i.e., thresholds in lmart (why not in tree).
    // used in roothistogram.
    float* get_fvector(const unsigned int i) const { return features[i]; }
    void set_fvector(const unsigned int i, float* v) { features[i] = v; }

    float** get_fmatrix() const { return features; }
    void set_fmatrix(float** m) { features=m; }

    unsigned int get_rloffsets(unsigned int i) const { return rloffsets[i]; }
    void set_rloffsets_vector( unsigned int* o) { rloffsets=o; }
    void set_rloffsets(const unsigned int i, const unsigned int o) { rloffsets[i]=o; }

    // used to find uniq values, i.e., thresholds in lmart (why not in tree)
    void sort_dpbyfeature(unsigned int i, unsigned int *&sorted, unsigned int &sortedsize) {
      sortedsize = ndps;
      sorted = idxfloat_radixsort(features[i], sortedsize);
    }
    /* CLA: this is not used
    double get_label(unsigned int i) const {
      return labels[i];
    }*/

    unsigned int get_featureid(unsigned int fidx) const { return usedfid[fidx]; }
    void set_featureid(unsigned int fidx, unsigned int val) { usedfid[fidx] = val; }
    void set_featureid_vector( unsigned int* v) { usedfid = v; }

    void set_rlids_vector( int* v) { rlids = v; }
    void set_rlids(const int i, const int v) { rlids[i] = v; }

    void set_labels_vector( double* v) { labels = v; }
    void set_label(const unsigned int i, const double v) { labels[i] = v; }

    /* CLA: this is not used
    unsigned int get_maxrlsize() const {
      return maxrlsize;
    }*/
  private:
    unsigned int nrankedlists = 0, ndps = 0, nfeatures = 0, maxrlsize = 0;
    unsigned int *rloffsets = NULL; //[0..nrankedlists] i-th rankedlist begins at rloffsets[i] and ends at rloffsets[i+1]-1
    double *labels = NULL; //[0..ndps-1]
    float **features = NULL; //[0..maxfid][0..ndps-1]
    int *rlids = NULL; //[0..nrankedlists-1]
    unsigned int *usedfid = NULL; //
    #ifndef SKIP_DPDESCRIPTION
    char **descriptions = NULL; //[0..ndps-1]
    #endif
};

#endif
